# ONNX Runtime Language Energy Study

Empirical study measuring energy consumption of ONNX Runtime inference across 7 host languages, 5 models, 5 inference volumes, and 2 execution phases. The runtime (ORT) is held constant across all runs; only the host language binding changes.

**Design:** 7 languages × 5 models × 5 volumes × 2 phases × 5 reps = **1,750 runs**

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
| Ruby     | Community (`ankane/onnxruntime`) |

| Model | Task | Parameters |
|-------|------|-----------|
| `all-MiniLM-L6-v2` | Sentence embedding | 22.7 M |
| `ModernBERT-base` | Fill-mask | 149 M |
| `gpt2` | Text generation (20-token greedy) | 117 M |
| `vit-face-expression` | Image classification | 85.8 M |
| `fashion-clip` | Zero-shot image classification | 200 M |

Inference volumes: 1, 10, 100, 1 000, 10 000 sequential calls per run.
Phases: **cold** (measurement includes session init) and **steady** (300 warmup calls, then measure).

---

## Prerequisites

### System packages

```bash
sudo apt install build-essential default-jdk maven ruby perf jq util-linux
```

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
3. Downloads all 5 ONNX models to `models/` via `scripts/download_models.py`
4. Prepares input tensors to `inputs/` via `scripts/prepare_inputs.py`
5. Builds C, C++, Java (Maven), C# (.NET), and Rust binaries; installs Ruby gem
6. Runs `measure.sh` — the full 1,750-run experiment

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
  "resources": { "peak_mem_mb",      -- VmPeak from /proc/<pid>/status
                 "cpu_util_pct"      -- from perf stat task-clock }
}
```

All RAPL readings are taken by a single Python wrapper (`scripts/measure.py`) that synchronises with each language runner via a pipe handshake, eliminating per-language discrepancies in energy measurement.

The run order is generated once, with repetition blocks emitted in `1, 2, 3, 4, 5` order and randomised within each block, then stored in `data/run_order.json`. Re-running `measure.sh` reuses the same order, so partial reruns are consistent.

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
  gen_run_order.py      # generates randomised run_order.json
  inference/            # language runners — inference only, no energy measurement
    python/infer.py
    c/                  # infer.c + Makefile
    cpp/                # infer.cpp + Makefile
    java/               # Maven project (pom.xml targets Java 11)
    csharp/OnnxInference/
    rust/               # Cargo.toml; links lib/onnxruntime/lib/libonnxruntime.so
    ruby/infer.rb
  analysis/             # post-hoc analysis scripts
models/                 # downloaded ONNX files (created by download_models.py)
inputs/                 # .npy input tensors (created by prepare_inputs.py)
lib/onnxruntime/        # shared ORT native library used by C, C++, Rust, Ruby
data/
  run_order.json        # generated once; defines the 1750-run order
  measurements.json     # results (appended per run)
logs/                   # per-run stderr logs (optional)
```

---

## Hardware and OS

The study was conducted on:

- **CPU:** Intel Core i5 (12th Gen)
- **RAM:** 16 GB
- **OS:** Ubuntu Linux (x86-64)
- **Java:** OpenJDK 11
- **ORT version:** 1.21.0

`measure.sh` sets the CPU frequency governor to `performance` and disables Intel Turbo Boost before the experiment, restoring the original state on exit.
