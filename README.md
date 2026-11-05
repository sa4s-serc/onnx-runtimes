# ONNX Runtime Language Energy Study

Empirical study measuring energy consumption of ONNX Runtime inference across 7 host languages, 6 models, 4 inference volumes, and 2 execution phases. The runtime (ORT) is held constant across all runs; only the host language binding changes. Ruby is measured in three interpreter configurations (3.0.2 without JIT, and 3.4.9 and 4.0.5 with YJIT), giving 9 language configurations.

**Design:** 9 language configurations × 6 models × 4 volumes × 2 phases × 5 reps = **2,160 runs**

---

## Languages and Models

| Language | Binding |
|----------|---------|
| Python   | Official |
| C        | Official |
| C++      | Official |
| Java     | Official |
| C#       | Official |
| Rust     | Community (`ort` crate) |
| Ruby     | Community (`ankane/onnxruntime`); labels `ruby` (3.0.2), `ruby34` (3.4.9, YJIT), `ruby40` (4.0.5, YJIT) |

| Model | Task | Parameters |
|-------|------|-----------|
| `all-MiniLM-L6-v2` | Sentence embedding | 22.7 M |
| `ModernBERT-base` | Fill-mask | 149 M |
| `gpt2` | Text generation (20-token greedy) | 117 M |
| `pythia-160m` | Text generation (20-token greedy) | 162 M |
| `vit-face-expression` | Image classification | 85.8 M |
| `fashion-clip` | Zero-shot image classification | 200 M |

Inference volumes: 1, 10, 100, 1 000 sequential calls per run.
Phases: **cold** (measurement includes session init) and **steady** (300 warmup calls, then measure).

---

## Prerequisites

### System packages

```bash
sudo apt install build-essential openjdk-21-jdk maven ruby perf jq util-linux
```

The Java runner targets Java 21 (`pom.xml`). The system `ruby` package provides the `ruby` (3.0.2) configuration. The `ruby34` and `ruby40` configurations expect rbenv installs at `~/.rbenv/versions/3.4.9` and `~/.rbenv/versions/4.0.5`, each with the `onnxruntime` gem, and are run with `--yjit`.

`perf` may require the kernel tools package matching your kernel:

```bash
sudo apt install linux-tools-generic linux-tools-$(uname -r)
```

### Optional runtimes (required to build those language binaries)

**Rust:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**.NET SDK (C#):**
Follow the instructions at https://dot.net/ — install .NET 8.

### RAPL permissions

Energy measurement via Intel RAPL requires read access to `/sys/class/powercap/intel-rapl`. `measure.sh` handles this automatically if `sudo` is available. All RAPL reads are performed by `scripts/measure.py` — a single Python process that wraps every language runner — ensuring identical measurement code across all languages.

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```

Without `sudo` the script checks that the files are already readable and continues; if not, it exits with an error message.

---

## Replication

Run **one command** from the repository root:

```bash
bash replicate.sh
```

This performs every step in order:

1. Creates a Python virtual environment (`.venv/`)
2. Installs Python packages: `onnxruntime==1.21.0`, `numpy`, `datasets`, `transformers`, `huggingface_hub`, `pillow`
3. Downloads all 6 ONNX models to `models/` via `scripts/download_models.py`
4. Prepares input tensors to `inputs/` via `scripts/prepare_inputs.py`
5. Builds C, C++, Java (Maven), C# (.NET), and Rust binaries; installs Ruby gem
6. Runs `measure.sh` over the runs listed in `data/run_order.json` (see [Run order](#run-order))

The script is **fail-fast**: any step failure aborts immediately. Missing optional runtimes (Rust, .NET) produce a clear error.

---

## Running measurements only

If the environment is already set up (venv exists, models and inputs downloaded, binaries built):

```bash
bash measure.sh
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--core N` | `2` | CPU core to pin inference to (`taskset -c N`) |
| `--sleep N` | `15` | Seconds between runs (thermal stabilisation) |
| `--languages L1,L2,...` | all | Comma-separated subset (e.g. `python,c,java`) |
| `--dry-run` | off | Print commands without executing |

`measure.sh` supports **resuming**: runs whose `run_id` already appears in `data/measurements.json` are skipped, so a partial run can be continued safely.

---

## Output

Results are appended to `data/measurements.json` — one JSON object per completed run.

Each object schema:

```
{
  "run_id", "timestamp_ist", "run_order", "repetition",
  "treatment": { "language", "model_id", "volume", "phase",
                 "warmup_count", "rapl_repetitions" },
  "energy":    { "cpu_pkg_j",        -- total CPU package energy (J), RAPL PKG domain
                 "init_cpu_j",       -- cold only: session-load energy (J)
                 "inference_cpu_j"   -- cold only: inference energy (J) },
  "time":      { "total_ms", "init_ms", "inference_ms" },
  "resources": { "peak_mem_mb",      -- max VmRSS from /proc/<pid>/status, sampled every 100 ms
                 "cpu_util_pct"      -- from perf stat task-clock }
}
```

All RAPL readings are taken by a single Python wrapper (`scripts/measure.py`) that synchronises with each language runner via a pipe handshake, eliminating per-language discrepancies in energy measurement.

### Run order

The campaign ran in two parts, and each has its own run-order file. Within each file, repetition blocks are emitted in `1, 2, 3, 4, 5` order and randomised within each block.

| File | Runs | Contents |
|------|------|----------|
| `data/run_order_original.json` | 1–1,400 | 7 languages (Ruby 3.0.2 only) × 5 models (no Pythia); from `gen_run_order.py` |
| `data/run_order.json` | 1,401–2,160 | `ruby34` and `ruby40` × 6 models, plus the 7 original languages × Pythia; from `gen_run_order_extra.py` |

`measure.sh` always reads `data/run_order.json`. To repeat the whole campaign from scratch, move `data/measurements.json` aside, copy `run_order_original.json` to `run_order.json` and run `measure.sh`, then restore the 760-run file and run `measure.sh` again. `run_order_extra.json` and `run_order_full.json` are earlier plans for the second part. None of their new runs were executed (`run_order_full.json` contains the original 1,400 runs plus 400 unexecuted ones).

---

## Repository layout

```
replicate.sh            # end-to-end replication entry point
measure.sh              # experiment orchestrator (CPU governor, perf stat, resume)
scripts/
  measure.py            # central RAPL/energy wrapper — called once per run;
                        #   launches the language runner, reads RAPL at the right
                        #   moment via a pipe handshake, and emits the result JSON
  download_models.py    # downloads ONNX models from Hugging Face Hub
  prepare_inputs.py     # generates input tensors (.npy) for each model × volume
  gen_run_order.py      # generates the run order for runs 1-1,400
  gen_run_order_extra.py # generates the run order for runs 1,401-2,160
  resume_campaign.sh    # @reboot cron helper that resumes measure.sh
  inference/            # language runners — inference only, no energy measurement
    python/infer.py
    c/                  # infer.c + Makefile
    cpp/                # infer.cpp + Makefile
    java/               # Maven project (pom.xml targets Java 21)
    csharp/OnnxInference/
    rust/               # Cargo.toml; links lib/onnxruntime/lib/libonnxruntime.so
    ruby/infer.rb
  analysis/             # post-hoc analysis scripts
models/                 # downloaded ONNX files (created by download_models.py)
inputs/                 # .npy input tensors (created by prepare_inputs.py)
lib/onnxruntime/        # shared ORT native library used by C, C++, Rust, Ruby
data/
  run_order_original.json  # runs 1-1,400 (first part of the campaign)
  run_order.json        # runs 1,401-2,160 (second part); read by measure.sh
  measurements.json     # results for all 2,160 runs (appended per run)
logs/                   # per-run stderr logs (optional)
```

---

## Hardware and OS

The study was conducted on:

- **Machine:** Dell OptiPlex Tower 7020
- **CPU:** Intel Core i5-12500 (6 cores, 12 threads)
- **RAM:** 8 GiB DDR5
- **OS:** Ubuntu 22.04.5 LTS, kernel 6.8.0-107
- **ORT version:** 1.21.0 (CPU execution provider)
- **C / C++:** GCC 11.4.0, `-O2`
- **Python:** 3.11.15
- **Java:** OpenJDK 21.0.7 (Temurin)
- **.NET:** SDK 8.0.125 / runtime 8.0.25
- **Rust:** 1.94.1, `--release`
- **Ruby:** 3.0.2 (no JIT); 3.4.9 and 4.0.5 (YJIT)

`measure.sh` sets the CPU frequency governor to `performance` and disables Intel Turbo Boost before the experiment, restoring the original state on exit.
