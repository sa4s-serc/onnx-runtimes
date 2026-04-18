#!/usr/bin/env python3
"""
Full statistical analysis for the ONNX host-language overhead study.
Produces all numbers, tables, and figures needed for the paper.
"""

import json
import warnings
from pathlib import Path
from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, kendalltau, shapiro, linregress

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "measurements.json"
OUT  = ROOT / "scripts" / "analysis"

LANG_ORDER = ["c", "cpp", "rust", "csharp", "java", "python", "ruby"]
LANG_DISPLAY = {
    "c": "C", "cpp": "C++", "rust": "Rust",
    "csharp": "C#", "java": "Java", "python": "Python", "ruby": "Ruby"
}
MODEL_SHORT = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
    "answerdotai/ModernBERT-base": "ModernBERT",
    "openai-community/gpt2": "GPT-2",
    "trpakov/vit-face-expression": "ViT",
    "patrickjohncyh/fashion-clip": "CLIP",
}
MODEL_ORDER  = ["MiniLM", "ModernBERT", "GPT-2", "ViT", "CLIP"]
# Distinct, colorblind-friendly model palette for grouped bar charts.
MODEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

# Cliff's delta thresholds
def cliffs_delta(x, y):
    x, y = np.array(x), np.array(y)
    n = len(x) * len(y)
    greater = sum(1 for xi in x for yi in y if xi > yi)
    less    = sum(1 for xi in x for yi in y if xi < yi)
    return (greater - less) / n

def cliffs_magnitude(d):
    d = abs(d)
    if d < 0.147: return "negligible"
    if d < 0.330: return "small"
    if d < 0.474: return "medium"
    return "large"

# ── load data ──────────────────────────────────────────────────────────────────
with open(DATA) as f:
    raw = json.load(f)

df = pd.json_normalize(raw)
df.columns = [c.replace("treatment.", "").replace("energy.", "")
               .replace("time.", "").replace("resources.", "") for c in df.columns]
df["model_short"] = df["model_id"].map(MODEL_SHORT)

print(f"Loaded {len(df)} runs")
print(f"Languages: {sorted(df['language'].unique())}")
print(f"Models: {sorted(df['model_id'].unique())}")
print(f"Volumes: {sorted(df['volume'].unique())}")
print(f"Phases: {sorted(df['phase'].unique())}")
print()

# ── helpers ────────────────────────────────────────────────────────────────────
def get_energy(lang, model_short, volume=None, phase="steady"):
    mask = (df["language"] == lang) & (df["model_short"] == model_short) & (df["phase"] == phase)
    if volume is not None:
        mask &= (df["volume"] == volume)
    return df[mask]["cpu_pkg_j"].values

def median_iqr(arr):
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Per-language median energy (steady-state, V=100, all models pooled)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TABLE: Steady-state median energy (J) at V=100, per language and model")
print("=" * 70)

rows = []
for lang in LANG_ORDER:
    row = {"Language": LANG_DISPLAY[lang]}
    for mod in MODEL_ORDER:
        vals = get_energy(lang, mod, volume=100, phase="steady")
        med, q1, q3 = median_iqr(vals)
        row[mod] = med
    rows.append(row)

tbl = pd.DataFrame(rows).set_index("Language")
print(tbl.round(3).to_string())
print()

# Overhead ratios relative to C at V=100 steady
print("Overhead ratios relative to C (V=100, steady-state):")
c_vals = {mod: np.median(get_energy("c", mod, volume=100, phase="steady")) for mod in MODEL_ORDER}
for lang in LANG_ORDER:
    ratios = []
    for mod in MODEL_ORDER:
        v = np.median(get_energy(lang, mod, volume=100, phase="steady"))
        ratios.append(v / c_vals[mod] if c_vals[mod] > 0 else np.nan)
    print(f"  {LANG_DISPLAY[lang]:8s}: {[f'{r:.3f}' for r in ratios]}  mean={np.nanmean(ratios):.3f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# RQ1: Kruskal-Wallis + pairwise Mann-Whitney per model (V=100, steady)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RQ1: Kruskal-Wallis test (language effect) per model, V=100, steady")
print("=" * 70)

rq1_results = {}
for mod in MODEL_ORDER:
    groups = [get_energy(lang, mod, volume=100, phase="steady") for lang in LANG_ORDER]
    stat, p = kruskal(*groups)
    rq1_results[mod] = {"stat": stat, "p": p}
    print(f"  {mod:12s}: H={stat:.2f}, p={p:.4e}  {'*' if p < 0.05 else ''}")

print()
print("Pairwise Mann-Whitney U (Holm-corrected, V=100, steady):")

# Collect all pairs across all models then apply Holm
pair_results = []
for mod in MODEL_ORDER:
    pairs = list(combinations(LANG_ORDER, 2))
    for la, lb in pairs:
        xa = get_energy(la, mod, volume=100, phase="steady")
        xb = get_energy(lb, mod, volume=100, phase="steady")
        if len(xa) == 0 or len(xb) == 0:
            continue
        stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
        d = cliffs_delta(xa, xb)
        pair_results.append({
            "model": mod, "lang_a": la, "lang_b": lb,
            "U": stat, "p_raw": p, "delta": d,
            "magnitude": cliffs_magnitude(d)
        })

# Holm correction within each model
pairwise_df = pd.DataFrame(pair_results)
for mod in MODEL_ORDER:
    mask = pairwise_df["model"] == mod
    sub = pairwise_df[mask].copy()
    ranked = sub.sort_values("p_raw")
    n = len(ranked)
    p_adj = []
    for i, (_, row) in enumerate(ranked.iterrows()):
        p_adj.append(min(1.0, row["p_raw"] * (n - i)))
    ranked["p_holm"] = p_adj
    pairwise_df.loc[mask, "p_holm"] = ranked["p_holm"].values
    sig = ranked[ranked["p_holm"] < 0.05]
    print(f"  {mod}: {len(sig)}/{len(ranked)} pairs significant after Holm correction")
    # Show large effects
    large = ranked[ranked["magnitude"].isin(["large", "medium"])]
    for _, r in large.head(5).iterrows():
        print(f"    {LANG_DISPLAY[r['lang_a']]} vs {LANG_DISPLAY[r['lang_b']]}: "
              f"delta={r['delta']:.3f} ({r['magnitude']}), p_holm={r['p_holm']:.4f}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# RQ2: Ranking consistency across models (Kendall's tau)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RQ2: Ranking stability across models (Kendall's tau, V=100, steady)")
print("=" * 70)

# Rank languages by median energy per model
rankings = {}
for mod in MODEL_ORDER:
    meds = [(lang, np.median(get_energy(lang, mod, volume=100, phase="steady")))
            for lang in LANG_ORDER]
    meds.sort(key=lambda x: x[1])
    rankings[mod] = [lang for lang, _ in meds]
    print(f"  {mod:12s}: {' < '.join(LANG_DISPLAY[l] for l in rankings[mod])}")

# Kendall's tau between all pairs of models
print("\n  Kendall's tau between model rankings:")
model_pairs = list(combinations(MODEL_ORDER, 2))
taus = []
for ma, mb in model_pairs:
    ra = [LANG_ORDER.index(l) for l in rankings[ma]]
    rb = [LANG_ORDER.index(l) for l in rankings[mb]]
    tau, p = kendalltau(ra, rb)
    taus.append(tau)
    print(f"    {ma} vs {mb}: tau={tau:.3f} (p={p:.4f})")
print(f"  Mean tau across all model pairs: {np.mean(taus):.3f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# RQ3: Energy scaling with volume (linear regression per language)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RQ3: Energy scaling with inference volume (steady-state, per language)")
print("=" * 70)

volumes = sorted(int(v) for v in df["volume"].dropna().unique())

# Pool across all models for a language-level picture
print("\nPooled across all models:")
lang_fits = {}
for lang in LANG_ORDER:
    all_x, all_y = [], []
    for vol in volumes:
        for mod in MODEL_ORDER:
            vals = get_energy(lang, mod, volume=vol, phase="steady")
            all_x.extend([vol] * len(vals))
            all_y.extend(vals)
    slope, intercept, r, p, se = linregress(all_x, all_y)
    lang_fits[lang] = {"slope": slope, "intercept": intercept, "r2": r**2}
    print(f"  {LANG_DISPLAY[lang]:8s}: slope={slope:.4f} J/inf, "
          f"intercept={intercept:.4f} J, R²={r**2:.4f}")

# Crossover volumes between C and each language
print("\nCrossover volumes (volume at which energy gap becomes practical):")
for lang in LANG_ORDER:
    if lang == "c":
        continue
    d_intercept = lang_fits[lang]["intercept"] - lang_fits["c"]["intercept"]
    d_slope     = lang_fits["c"]["slope"]     - lang_fits[lang]["slope"]
    if abs(d_slope) < 1e-8:
        print(f"  C vs {LANG_DISPLAY[lang]:8s}: parallel slopes, no crossover")
    else:
        v_star = d_intercept / d_slope
        print(f"  C vs {LANG_DISPLAY[lang]:8s}: V* = {v_star:.1f} inferences")

print()

# Per-model scaling (slopes)
print("Per-language per-inference slope by model (J per inference, steady):")
model_slopes = {}
for mod in MODEL_ORDER:
    slopes = {}
    for lang in LANG_ORDER:
        x = []
        y = []
        for vol in volumes:
            vals = get_energy(lang, mod, volume=vol, phase="steady")
            x.extend([vol] * len(vals))
            y.extend(vals)
        slope, intercept, r, p, se = linregress(x, y)
        slopes[lang] = slope
    model_slopes[mod] = slopes
    line = "  " + f"{mod:12s}: "
    line += "  ".join(f"{LANG_DISPLAY[l]}={slopes[l]:.4f}" for l in LANG_ORDER)
    print(line)

print()

# ═══════════════════════════════════════════════════════════════════════════════
# RQ4: Cold-start vs. steady-state
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RQ4: Cold-start initialisation energy (V=1, per language × model)")
print("=" * 70)

print("\nInit energy fraction (init_cpu_j / cpu_pkg_j) for cold-start V=1:")
for lang in LANG_ORDER:
    fracs = []
    for mod in MODEL_ORDER:
        mask = (df["language"] == lang) & (df["model_short"] == mod) & \
               (df["phase"] == "cold") & (df["volume"] == 1)
        sub = df[mask]
        if len(sub) == 0:
            continue
        frac = (sub["init_cpu_j"] / sub["cpu_pkg_j"]).values
        fracs.extend(frac)
    print(f"  {LANG_DISPLAY[lang]:8s}: mean init fraction = {np.mean(fracs):.3f} "
          f"(range {np.min(fracs):.3f}–{np.max(fracs):.3f})")

print("\nCold vs steady energy ratio (cold V=1 / steady V=1) per language:")
for lang in LANG_ORDER:
    ratios = []
    for mod in MODEL_ORDER:
        cold   = get_energy(lang, mod, volume=1, phase="cold")
        steady = get_energy(lang, mod, volume=1, phase="steady")
        if len(cold) > 0 and len(steady) > 0:
            ratios.append(np.median(cold) / np.median(steady))
    print(f"  {LANG_DISPLAY[lang]:8s}: median cold/steady ratio = {np.median(ratios):.2f} "
          f"(across models)")

print("\nMann-Whitney U cold vs steady (V=1000) per language × model:")
for lang in LANG_ORDER:
    for mod in MODEL_ORDER:
        cold   = get_energy(lang, mod, volume=1000, phase="cold")
        steady = get_energy(lang, mod, volume=1000, phase="steady")
        if len(cold) == 0 or len(steady) == 0:
            continue
        stat, p = mannwhitneyu(cold, steady, alternative="two-sided")
        d = cliffs_delta(cold, steady)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {LANG_DISPLAY[lang]:8s}/{mod:12s}: delta={d:.3f} "
              f"({cliffs_magnitude(d)}), p={p:.4f} {sig}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# LATENCY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LATENCY: Median steady-state inference time (ms) at V=100 per language")
print("=" * 70)

for lang in LANG_ORDER:
    meds = []
    for mod in MODEL_ORDER:
        mask = (df["language"] == lang) & (df["model_short"] == mod) & \
               (df["phase"] == "steady") & (df["volume"] == 100)
        sub = df[mask]
        if len(sub) > 0:
            meds.append(np.median(sub["inference_ms"]))
    if meds:
        print(f"  {LANG_DISPLAY[lang]:8s}: {[f'{m:.1f}' for m in meds]}  mean={np.mean(meds):.1f} ms")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("MEMORY: Median peak RSS (MB) at V=100, steady-state")
print("=" * 70)

for lang in LANG_ORDER:
    meds = []
    for mod in MODEL_ORDER:
        mask = (df["language"] == lang) & (df["model_short"] == mod) & \
               (df["phase"] == "steady") & (df["volume"] == 100)
        sub = df[mask]
        if len(sub) > 0:
            meds.append(np.median(sub["peak_mem_mb"]))
    if meds:
        print(f"  {LANG_DISPLAY[lang]:8s}: {[f'{m:.0f}' for m in meds]}  mean={np.mean(meds):.0f} MB")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES for the paper
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LATEX TABLE: Median energy (J) ± IQR at V=100, steady-state")
print("=" * 70)

latex = []
latex.append(r"\begin{table}[t]")
latex.append(r"\caption{Median CPU package energy (J) and IQR across 5 repetitions, "
             r"steady-state, $V=100$. Bold = lowest per column.}")
latex.append(r"\label{tab:energy_v100}")
latex.append(r"\small")
latex.append(r"\begin{tabular}{l" + "r" * len(MODEL_ORDER) + "}")
latex.append(r"\toprule")
latex.append("Language & " + " & ".join(MODEL_ORDER) + r" \\")
latex.append(r"\midrule")

# find per-column minima
col_mins = {}
for mod in MODEL_ORDER:
    vals = [np.median(get_energy(l, mod, volume=100, phase="steady")) for l in LANG_ORDER]
    col_mins[mod] = min(v for v in vals if not np.isnan(v))

for lang in LANG_ORDER:
    cells = [LANG_DISPLAY[lang]]
    for mod in MODEL_ORDER:
        arr = get_energy(lang, mod, volume=100, phase="steady")
        med, q1, q3 = median_iqr(arr)
        val_str = f"{med:.2f}"
        if abs(med - col_mins[mod]) < 0.01:
            val_str = r"\textbf{" + val_str + "}"
        cells.append(val_str)
    latex.append(" & ".join(cells) + r" \\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

print("\n".join(latex))
print()

# Overhead ratio table
print("=" * 70)
print("LATEX TABLE: Energy overhead ratios relative to C")
print("=" * 70)

latex2 = []
latex2.append(r"\begin{table}[t]")
latex2.append(r"\caption{Mean energy overhead ratio relative to C across all models, "
              r"steady-state. Values $>1$ indicate higher energy than C.}")
latex2.append(r"\label{tab:overhead}")
latex2.append(r"\small")
latex2.append(r"\begin{tabular}{l" + "r" * len(MODEL_ORDER) + "r}")
latex2.append(r"\toprule")
latex2.append("Language & " + " & ".join(MODEL_ORDER) + r" & Mean \\" )
latex2.append(r"\midrule")

for lang in LANG_ORDER:
    cells = [LANG_DISPLAY[lang]]
    ratios = []
    for mod in MODEL_ORDER:
        c_med  = np.median(get_energy("c", mod, volume=100, phase="steady"))
        l_med  = np.median(get_energy(lang, mod, volume=100, phase="steady"))
        ratio  = l_med / c_med if c_med > 0 else np.nan
        ratios.append(ratio)
        cells.append(f"{ratio:.2f}$\\times$")
    cells.append(f"{np.nanmean(ratios):.2f}$\\times$")
    latex2.append(" & ".join(cells) + r" \\")

latex2.append(r"\bottomrule")
latex2.append(r"\end{tabular}")
latex2.append(r"\end{table}")
print("\n".join(latex2))
print()

# Cold-start summary table
print("=" * 70)
print("LATEX TABLE: Cold-start init energy fraction (V=1)")
print("=" * 70)

latex3 = []
latex3.append(r"\begin{table}[t]")
latex3.append(r"\caption{Median cold-start initialisation energy as a fraction of total run energy "
              r"($V=1$, i.e.\ the first inference of a fresh process). Higher fractions indicate "
              r"the binding imposes disproportionate session-creation overhead.}")
latex3.append(r"\label{tab:coldstart}")
latex3.append(r"\small")
latex3.append(r"\begin{tabular}{l" + "r" * len(MODEL_ORDER) + "}")
latex3.append(r"\toprule")
latex3.append("Language & " + " & ".join(MODEL_ORDER) + r" \\")
latex3.append(r"\midrule")

for lang in LANG_ORDER:
    cells = [LANG_DISPLAY[lang]]
    for mod in MODEL_ORDER:
        mask = (df["language"] == lang) & (df["model_short"] == mod) & \
               (df["phase"] == "cold") & (df["volume"] == 1)
        sub = df[mask]
        if len(sub) == 0:
            cells.append("---")
        else:
            frac = np.median((sub["init_cpu_j"] / sub["cpu_pkg_j"]).values)
            cells.append(f"{frac*100:.1f}\\%")
    latex3.append(" & ".join(cells) + r" \\")

latex3.append(r"\bottomrule")
latex3.append(r"\end{tabular}")
latex3.append(r"\end{table}")
print("\n".join(latex3))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES: produce matplotlib plots
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    COLORS = {
        "c": "#2196F3",       # blue
        "cpp": "#1565C0",     # dark blue
        "rust": "#FF5722",    # deep orange
        "csharp": "#9C27B0",  # purple
        "java": "#F44336",    # red
        "python": "#4CAF50",  # green
        "ruby": "#E91E63",    # pink
    }
    MARKERS = {
        "c": "o", "cpp": "s", "rust": "^", "csharp": "D",
        "java": "v", "python": "P", "ruby": "X"
    }

    # Figure 1: Energy vs volume, 2-row grid (3+2), steady-state
    fig, axes_grid = plt.subplots(2, 3, figsize=(7, 5.0), sharey=False)
    axes_flat = axes_grid.flatten()
    axes_flat[-1].set_visible(False)  # hide unused 6th cell

    for ax, mod in zip(axes_flat, MODEL_ORDER):
        for lang in LANG_ORDER:
            meds = [np.median(get_energy(lang, mod, vol, "steady")) for vol in volumes]
            ax.plot(volumes, meds,
                    color=COLORS[lang], marker=MARKERS[lang],
                    label=LANG_DISPLAY[lang], linewidth=1.5, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(mod, fontsize=9)
        ax.set_xlabel("Inferences", fontsize=8)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(volumes)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylabel("CPU energy (J)", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right",
               bbox_to_anchor=(0.98, 0.05), ncol=1, fontsize=7.5,
               framealpha=0.85)
    plt.tight_layout()
    plt.savefig(OUT / "fig_energy_vs_volume.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_energy_vs_volume.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_energy_vs_volume.pdf/png")

    # Figure 2: Cold vs steady per language (all available volumes, averaged across models)
    n_vols = len(volumes)
    ncols = min(4, n_vols)
    nrows = math.ceil(n_vols / ncols)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.3 * nrows), squeeze=False)
    axes_flat = axes_grid.flatten()

    for ax, vol in zip(axes_flat, volumes):
        cold_meds   = []
        steady_meds = []
        for lang in LANG_ORDER:
            vals_cold   = [np.median(get_energy(lang, mod, vol, "cold"))   for mod in MODEL_ORDER]
            vals_steady = [np.median(get_energy(lang, mod, vol, "steady")) for mod in MODEL_ORDER]
            cold_meds.append(np.nanmean(vals_cold))
            steady_meds.append(np.nanmean(vals_steady))
        x = np.arange(len(LANG_ORDER))
        w = 0.35
        ax.bar(x - w/2, cold_meds,   w, label="Cold start",   color="#EF5350")
        ax.bar(x + w/2, steady_meds, w, label="Steady state", color="#42A5F5")
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_DISPLAY[l] for l in LANG_ORDER], rotation=30, ha="right")
        ax.set_title(f"V={vol}")
        ax.set_ylabel("Mean energy (J)")
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat[n_vols:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT / "fig_cold_vs_steady.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_cold_vs_steady.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_cold_vs_steady.pdf/png")

    # Figure 3: Overhead bar chart (grouped by language, one bar per model)
    plot_lang_display = {k: v.replace("\\#", "#") for k, v in LANG_DISPLAY.items()}
    ratio_data = {}
    for lang in LANG_ORDER:
        ratios = []
        for mod in MODEL_ORDER:
            c_med = np.median(get_energy("c", mod, volume=100, phase="steady"))
            l_med = np.median(get_energy(lang, mod, volume=100, phase="steady"))
            ratios.append(l_med / c_med if c_med > 0 else np.nan)
        ratio_data[lang] = ratios

    fig, ax = plt.subplots(figsize=(7, 3.5))
    n_lang = len(LANG_ORDER)
    n_mod  = len(MODEL_ORDER)
    x      = np.arange(n_lang)
    width  = 0.15
    offsets = np.linspace(-(n_mod - 1) / 2, (n_mod - 1) / 2, n_mod) * width

    for j, (mod, color) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
        vals = [ratio_data[lang][j] for lang in LANG_ORDER]
        ax.bar(x + offsets[j], vals, width, label=mod, color=color,
               alpha=0.88, edgecolor="white", linewidth=0.4)

    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.55, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([plot_lang_display[l] for l in LANG_ORDER], fontsize=10)
    ax.set_ylabel("Energy overhead relative to C (×)", fontsize=10)
    ax.set_ylim(0, 4.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.legend(fontsize=8.5, ncol=5, loc="upper left", framealpha=0.85, columnspacing=0.8)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "fig_overhead_bar.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_overhead_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_overhead_bar.pdf/png")

    print("\nAll figures saved to", OUT)

except ImportError as e:
    print(f"Matplotlib not available: {e}")

print("\nAnalysis complete.")
