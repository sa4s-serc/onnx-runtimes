#!/usr/bin/env python3
"""
Equivalence analysis for the six-language cluster (RQ1 / reviewer 15C).

Addresses the omnibus-significant / no-pairwise-survives tension by replacing
"no significant difference" (absence of evidence) with positive, quantified
equivalence evidence:

  (1) TOST (two one-sided t-tests) per model at V=100 steady, equivalence
      margin +/-10% of the C reference, for each cluster language vs C.
  (2) Pooled overhead-ratio confidence intervals: for each language, the
      per-cell ratio median(lang)/median(C) across all steady (model, volume)
      cells, summarised as a geometric mean with a bootstrap 95% CI. This
      raises the effective sample size well above n=5 and yields a tight
      bound on the true overhead.

Uses only the existing measurement data (no new runs).
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "measurements.json"

CLUSTER = ["cpp", "rust", "csharp", "java", "python"]   # vs C reference
DISPLAY = {"c": "C", "cpp": "C++", "rust": "Rust", "csharp": "C#",
           "java": "Java", "python": "Python", "ruby": "Ruby"}
MODEL_SHORT = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
    "answerdotai/ModernBERT-base": "ModernBERT",
    "openai-community/gpt2": "GPT-2",
    "trpakov/vit-face-expression": "ViT",
    "patrickjohncyh/fashion-clip": "CLIP",
    "EleutherAI/pythia-160m": "Pythia",
}
MODELS = ["MiniLM", "ModernBERT", "GPT-2", "Pythia", "ViT", "CLIP"]
VOLUMES = [1, 10, 100, 1000]
MARGIN = 0.10          # +/-10% practical-equivalence margin
ALPHA = 0.05
RNG = np.random.default_rng(7)

raw = json.load(open(DATA))
def energy(lang, model, volume, phase="steady"):
    return np.array([r["energy"]["cpu_pkg_j"] for r in raw
                     if r["treatment"]["language"] == lang
                     and MODEL_SHORT.get(r["treatment"]["model_id"]) == model
                     and r["treatment"]["volume"] == volume
                     and r["treatment"]["phase"] == phase], dtype=float)


def tost_two_sample(a, b, margin_abs):
    """TOST for equivalence of mean(a)-mean(b) within +/-margin_abs (Welch)."""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    if se == 0:
        return diff, (abs(diff) < margin_abs), 0.0, 0.0
    dof = se**4 / ((a.var(ddof=1)/len(a))**2/(len(a)-1)
                   + (b.var(ddof=1)/len(b))**2/(len(b)-1))
    # H0: diff <= -margin  (upper one-sided)
    t_lower = (diff + margin_abs) / se
    p_lower = 1 - stats.t.cdf(t_lower, dof)
    # H0: diff >= +margin  (lower one-sided)
    t_upper = (diff - margin_abs) / se
    p_upper = stats.t.cdf(t_upper, dof)
    p_tost = max(p_lower, p_upper)
    return diff, (p_tost < ALPHA), p_tost, dof


print("=" * 74)
print("(1) TOST per model, V=100 steady, equivalence margin +/-10% of C")
print("=" * 74)
print(f"{'lang':7}" + "".join(f"{m:>13}" for m in MODELS))
tost_pass = {lang: 0 for lang in CLUSTER}
for lang in CLUSTER:
    row = f"{DISPLAY[lang]:7}"
    for m in MODELS:
        c = energy("c", m, 100)
        x = energy(lang, m, 100)
        if len(c) < 2 or len(x) < 2:
            row += f"{'n/a':>13}"; continue
        margin_abs = MARGIN * np.median(c)
        _, eq, p, _ = tost_two_sample(x, c, margin_abs)
        tost_pass[lang] += int(eq)
        row += f"{('EQ' if eq else '--')+f' p={p:.2f}':>13}"
    print(row)
nmods = len(MODELS)
print("\n  EQ = equivalence to C established at alpha=0.05 (both one-sided tests reject)")
print(f"  cells passing / {nmods}:", {DISPLAY[l]: f"{tost_pass[l]}/{nmods}" for l in CLUSTER})

print()
print("=" * 74)
print("(2) Pooled overhead-ratio vs C across all steady (model x volume) cells")
print(f"    geometric mean ratio with bootstrap 95% CI  (n_cells = {len(MODELS)} x {len(VOLUMES)} = {len(MODELS)*len(VOLUMES)})")
print("=" * 74)
for lang in CLUSTER + ["ruby"]:
    ratios = []
    for m in MODELS:
        for v in VOLUMES:
            c = energy("c", m, v); x = energy(lang, m, v)
            if len(c) and len(x):
                ratios.append(np.median(x) / np.median(c))
    ratios = np.array(ratios)
    logr = np.log(ratios)
    gmean = float(np.exp(logr.mean()))
    # bootstrap CI on geometric mean
    boot = [np.exp(RNG.choice(logr, len(logr), replace=True).mean())
            for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    within = "within +/-10%" if hi <= 1.10 and lo >= 0.90 else \
             ("<=+10% upper" if hi <= 1.10 else "EXCEEDS 10%")
    print(f"  {DISPLAY[lang]:7} gmean={gmean:5.3f}x  95% CI [{lo:5.3f}, {hi:5.3f}]  "
          f"max_cell={ratios.max():.2f}x   {within}")

print()
print("Interpretation: for the six-language cluster the CI upper bound quantifies")
print("the worst-case true overhead; Ruby is reported as the contrast (wide, >1).")
