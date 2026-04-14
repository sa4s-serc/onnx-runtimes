#!/usr/bin/env python3
"""
Central energy/timing measurement wrapper for the ONNX language comparison study.

All RAPL energy readings are done here, ensuring identical measurement code for
every language. Each language runner handles only model loading and inference,
then signals this wrapper at the measurement boundary.

Usage:
    python measure.py <language> <model_short> <volume> <phase>
                      <repetition> <run_order> <run_id>

READY protocol (both phases):
  Runner prints "READY\\n" (flushed stdout) at the start of the measurement window,
  then blocks reading 1 byte from stdin. This wrapper reads RAPL at that moment,
  then sends "\\n" to unblock the runner.

  Cold phase:  measurement window = session load + inference
               rapl0 is read BEFORE launching runner
               READY is printed AFTER session load  →  rapl_mid split
               runner then runs K inferences and exits  →  rapl1

  Steady phase: measurement window = measured inference only (after warmup)
               runner loads model and runs warmup silently
               READY is printed after warmup  →  rapl0
               runner then runs R×K inferences and exits  →  rapl1

Runner stderr JSON:
    {"init_ms": <float|null>, "inference_ms": <float>, "total_ms": <float|null>}

    init_ms     : cold only — runner-measured ms from process start to session-ready
    inference_ms: runner-measured, already divided by R for steady state
    total_ms    : cold only — runner-measured ms from process start to inference end

peak_mem_mb is measured externally by this script via polling /proc/<pid>/status
(VmRSS sampled every 100 ms; peak across all samples reported as peak_mem_mb).
"""

import sys
import os
import json
import subprocess
import datetime
import threading
import time

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INPUTS_DIR = os.path.join(REPO_ROOT, "inputs")
LIB_ORT    = os.path.join(REPO_ROOT, "lib", "onnxruntime", "lib", "libonnxruntime.so")

MODEL_FILES = {
    "minilm":      "all-MiniLM-L6-v2.onnx",
    "modernbert":  "ModernBERT-base.onnx",
    "gpt2":        "gpt2.onnx",
    "vit":         "vit-face-expression.onnx",
    "fashionclip": "fashion-clip.onnx",
}

MODEL_IDS = {
    "minilm":      "sentence-transformers/all-MiniLM-L6-v2",
    "modernbert":  "answerdotai/ModernBERT-base",
    "gpt2":        "openai-community/gpt2",
    "vit":         "trpakov/vit-face-expression",
    "fashionclip": "patrickjohncyh/fashion-clip",
}

MODEL_EST_MS = {
    "minilm": 15, "modernbert": 80, "gpt2": 70, "vit": 80, "fashionclip": 100,
}

WARMUP_COUNT = 300
RAPL_PATH    = "/sys/class/powercap/intel-rapl:0/energy_uj"
MAX_RAPL_UJ  = 262_143_328_850.0   # counter wrap-around threshold
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# RAPL helpers
# ---------------------------------------------------------------------------

def _poll_peak_rss(pid: int, out: list, stop: threading.Event) -> None:
    """Background thread: poll VmRSS of pid every 100 ms, record peak in MB."""
    path = f"/proc/{pid}/status"
    peak_kb = 0
    while not stop.is_set():
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        if kb > peak_kb:
                            peak_kb = kb
                        break
        except FileNotFoundError:
            break
        except Exception:
            pass
        stop.wait(0.1)
    out.append(peak_kb / 1024.0)


def read_rapl_uj() -> int:
    try:
        with open(RAPL_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return -1


def rapl_diff_j(r0: int, r1: int) -> float | None:
    """Convert a RAPL start/end pair to joules, handling wrap-around."""
    if r0 < 0 or r1 < 0:
        return None
    uj = (r1 - r0) if r1 >= r0 else MAX_RAPL_UJ - (r0 - r1)
    return uj / 1_000_000.0


# ---------------------------------------------------------------------------
# Runner command factory
# ---------------------------------------------------------------------------

def runner_cmd(lang: str, model_path: str, inputs_dir: str,
               volume: int, phase: str, model_short: str) -> list[str]:
    """Return the argv list used to launch the language-specific runner."""
    runner_args = [model_path, inputs_dir, str(volume), phase, model_short]

    if lang == "python":
        script  = os.path.join(REPO_ROOT, "scripts", "inference", "python", "infer.py")
        venv_py = os.path.join(REPO_ROOT, ".venv", "bin", "python")
        py      = venv_py if os.path.exists(venv_py) else sys.executable
        return [py, script] + runner_args

    if lang == "c":
        binary = os.path.join(REPO_ROOT, "scripts", "inference", "c", "infer")
        return [binary] + runner_args

    if lang == "cpp":
        binary = os.path.join(REPO_ROOT, "scripts", "inference", "cpp", "infer")
        return [binary] + runner_args

    if lang == "ruby":
        script = os.path.join(REPO_ROOT, "scripts", "inference", "ruby", "infer.rb")
        return ["ruby", script] + runner_args

    if lang == "rust":
        binary = os.path.join(REPO_ROOT, "scripts", "inference", "rust",
                              "target", "release", "onnx_inference")
        return [binary] + runner_args

    if lang == "csharp":
        dll = os.path.join(REPO_ROOT, "scripts", "inference", "csharp",
                           "OnnxInference", "bin", "Release", "net8.0", "OnnxInference.dll")
        return ["dotnet", dll] + runner_args

    if lang == "java":
        jar = os.path.join(REPO_ROOT, "scripts", "inference", "java",
                           "target", "onnx-inference-1.0-SNAPSHOT.jar")
        return ["java", "-XX:CompileThreshold=100", "-jar", jar] + runner_args

    raise ValueError(f"Unknown language: {lang!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 8:
        print(
            f"Usage: {sys.argv[0]} <language> <model_short> <volume> <phase>"
            " <repetition> <run_order> <run_id>",
            file=sys.stderr,
        )
        sys.exit(1)

    language    = sys.argv[1]
    model_short = sys.argv[2]
    K           = int(sys.argv[3])
    phase       = sys.argv[4]
    repetition  = int(sys.argv[5])
    run_order   = int(sys.argv[6])
    run_id      = sys.argv[7]

    is_steady   = (phase == "steady")
    est_ms      = MODEL_EST_MS[model_short]
    R           = max(1, -(-(100) // (K * est_ms))) if is_steady else 1
    warmup      = WARMUP_COUNT if is_steady else 0
    model_path  = os.path.join(MODELS_DIR, MODEL_FILES[model_short])

    cmd = runner_cmd(language, model_path, INPUTS_DIR, K, phase, model_short)

    env = os.environ.copy()
    env.setdefault("ORT_DYLIB_PATH", LIB_ORT)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Start peak-RSS polling thread
    _rss_out: list[float] = []
    _rss_stop = threading.Event()
    _rss_thread = threading.Thread(
        target=_poll_peak_rss, args=(proc.pid, _rss_out, _rss_stop), daemon=True
    )
    _rss_thread.start()

    # ------------------------------------------------------------------
    # RAPL measurement using the READY protocol
    # ------------------------------------------------------------------
    rapl0 = rapl_mid = rapl1 = -1

    def raise_ready_error(line: bytes) -> None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        stderr_data = proc.stderr.read().decode(errors="replace").strip()
        detail = f"; stderr: {stderr_data}" if stderr_data else ""
        raise RuntimeError(f"Expected READY, got: {line!r}{detail}")

    if phase == "cold":
        # Start measuring from before the runner launches
        rapl0 = read_rapl_uj()

        # Wait for session-load to complete
        line = proc.stdout.readline()
        if line.strip() != b"READY":
            raise_ready_error(line)
        rapl_mid = read_rapl_uj()         # split: init / inference

        # Unblock runner → it will now run inference
        proc.stdin.write(b"\n")
        proc.stdin.flush()

        stderr_data = proc.stderr.read()
        proc.wait()
        rapl1 = read_rapl_uj()

    else:  # steady
        # Wait for warmup to finish; measurement starts here
        line = proc.stdout.readline()
        if line.strip() != b"READY":
            raise_ready_error(line)
        rapl0 = read_rapl_uj()            # measurement window opens

        # Unblock runner → it will now run R×K measured inferences
        proc.stdin.write(b"\n")
        proc.stdin.flush()

        stderr_data = proc.stderr.read()
        proc.wait()
        rapl1 = read_rapl_uj()            # measurement window closes

    _rss_stop.set()
    _rss_thread.join(timeout=1.0)
    peak_mem_mb = _rss_out[0] if _rss_out else 0.0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Runner exited {proc.returncode}; stderr: {stderr_data.decode(errors='replace')}"
        )

    # ------------------------------------------------------------------
    # Parse runner timing report
    # ------------------------------------------------------------------
    runner_info  = json.loads(stderr_data.decode().strip())
    init_ms      = runner_info.get("init_ms")       # float or None
    inference_ms = runner_info["inference_ms"]       # already ÷R for steady
    total_ms     = runner_info.get("total_ms")       # float or None (cold only)

    # ------------------------------------------------------------------
    # Compute energy
    # ------------------------------------------------------------------
    if phase == "cold":
        init_j  = rapl_diff_j(rapl0, rapl_mid)
        infer_j = rapl_diff_j(rapl_mid, rapl1)
        total_j = (init_j + infer_j) if (init_j is not None and infer_j is not None) else None
        # Fall back to runner-measured total_ms if not provided
        result_total_ms = total_ms if total_ms is not None else (
            (init_ms or 0.0) + inference_ms
        )
    else:
        raw_j   = rapl_diff_j(rapl0, rapl1)
        total_j = (raw_j / R) if raw_j is not None else None
        init_j  = None
        infer_j = None
        result_total_ms = inference_ms

    # ------------------------------------------------------------------
    # Assemble result JSON (same schema as before)
    # ------------------------------------------------------------------
    ts = datetime.datetime.now(IST).isoformat(timespec="seconds")

    def rnd6(v):
        return round(v, 6) if v is not None else None

    def rnd3(v):
        return round(v, 3) if v is not None else None

    result = {
        "run_id":        run_id,
        "timestamp_ist": ts,
        "run_order":     run_order,
        "repetition":    repetition,
        "treatment": {
            "language":         language,
            "model_id":         MODEL_IDS[model_short],
            "volume":           K,
            "phase":            phase,
            "warmup_count":     warmup,
            "rapl_repetitions": R,
        },
        "energy": {
            "cpu_pkg_j":       rnd6(total_j),
            "init_cpu_j":      rnd6(init_j),
            "inference_cpu_j": rnd6(infer_j),
        },
        "time": {
            "total_ms":     rnd3(result_total_ms),
            "init_ms":      rnd3(init_ms),
            "inference_ms": rnd3(inference_ms),
        },
        "resources": {
            "peak_mem_mb":  round(peak_mem_mb, 2),
            "cpu_util_pct": 0.0,
        },
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
