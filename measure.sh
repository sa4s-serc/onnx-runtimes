#!/usr/bin/env bash
# =============================================================================
# measure.sh — Single orchestration script for the ONNX language energy study.
#
# Handles:
#   - RAPL read permissions
#   - CPU governor and turbo-boost configuration
#   - Run-order generation (first run only)
#   - Per-language binary dispatch with taskset core pinning
#   - CPU utilisation measurement via perf stat
#   - Appending results to data/measurements.json
#   - Resume support (skips already-completed run_ids)
#   - Inter-run sleep for thermal stabilisation
#
# Usage:
#   bash measure.sh [--core N] [--sleep N] [--languages L1,L2,...] [--dry-run]
#
# Environment overrides (lower priority than CLI flags):
#   MEASURE_CORE=2     — CPU core to pin to (default: 2)
#   MEASURE_SLEEP=15   — seconds to sleep between runs (default: 15)
#
# Dependencies: bash, python3, jq, perf, taskset (util-linux), sudo
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
DATA_DIR="$REPO_ROOT/data"
SCRIPTS_DIR="$REPO_ROOT/scripts"
ORT_DYLIB="$REPO_ROOT/lib/onnxruntime/lib/libonnxruntime.so"

# Ruby gem path (user-installed gems)
RUBY_GEM_DIR="$(ruby -e 'puts Gem.user_dir' 2>/dev/null || echo '')"
if [[ -n "$RUBY_GEM_DIR" ]]; then
    export GEM_PATH="$RUBY_GEM_DIR${GEM_PATH:+:$GEM_PATH}"
    export PATH="$RUBY_GEM_DIR/bin${PATH:+:$PATH}"
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CORE="${MEASURE_CORE:-2}"
SLEEP_SECS="${MEASURE_SLEEP:-15}"
FILTER_LANGS=""     # empty = all languages
DRY_RUN=false

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --core)       CORE="$2";         shift 2 ;;
        --sleep)      SLEEP_SECS="$2";   shift 2 ;;
        --languages)  FILTER_LANGS="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;      shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[measure $(date '+%H:%M:%S')] $*"; }
die() { echo "[measure ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. RAPL read permissions
# ---------------------------------------------------------------------------
log "Granting RAPL read permissions..."
if ! sudo chmod -R a+r /sys/class/powercap/intel-rapl 2>/dev/null; then
    # sudo not available in this context; verify the files are already readable
    if ! cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1; then
        die "RAPL not readable. Run manually: sudo chmod -R a+r /sys/class/powercap/intel-rapl"
    fi
    log "  (sudo unavailable — RAPL already readable, continuing)"
fi

# ---------------------------------------------------------------------------
# 1. CPU frequency governor → performance
# ---------------------------------------------------------------------------
log "Setting CPU scaling governor to 'performance'..."
for gov_path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -f "$gov_path" ]] && echo "performance" | sudo tee "$gov_path" > /dev/null 2>&1 || true
done

# ---------------------------------------------------------------------------
# 2. Disable Intel Turbo Boost
# ---------------------------------------------------------------------------
TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"
if [[ -f "$TURBO_PATH" ]]; then
    log "Disabling Intel Turbo Boost (no_turbo=1)..."
    echo 1 | sudo tee "$TURBO_PATH" > /dev/null 2>&1 || log "  WARNING: could not disable Turbo Boost (run as root or grant sudo)"
fi

# ---------------------------------------------------------------------------
# 3. Sanity checks
# ---------------------------------------------------------------------------
[[ -f "$VENV_PY" ]]     || die "Python venv not found at $VENV_PY — run replicate.sh first"
command -v jq     &>/dev/null || die "jq not found — install with: sudo apt install jq"
command -v perf   &>/dev/null || die "perf not found — install with: sudo apt install linux-tools-generic"
command -v taskset &>/dev/null || die "taskset not found — install with: sudo apt install util-linux"

mkdir -p "$DATA_DIR"
LOGS_DIR="$REPO_ROOT/logs"
mkdir -p "$LOGS_DIR"

# ---------------------------------------------------------------------------
# 4. Generate run order if absent
# ---------------------------------------------------------------------------
RUN_ORDER_FILE="$DATA_DIR/run_order.json"
if [[ ! -f "$RUN_ORDER_FILE" ]]; then
    log "Generating randomised run order..."
    "$VENV_PY" "$SCRIPTS_DIR/gen_run_order.py"
fi

TOTAL_RUNS=$(jq 'length' "$RUN_ORDER_FILE")
log "Run order: $TOTAL_RUNS total runs loaded from $RUN_ORDER_FILE"

# ---------------------------------------------------------------------------
# 6. Initialise measurements.json if absent
# ---------------------------------------------------------------------------
MEAS_FILE="$DATA_DIR/measurements.json"
if [[ ! -f "$MEAS_FILE" ]]; then
    echo "[]" > "$MEAS_FILE"
fi

# Build set of already-completed run_ids for resume support
DONE_IDS_FILE=$(mktemp)
trap 'rm -f "$DONE_IDS_FILE"' EXIT
jq -r '.[].run_id' "$MEAS_FILE" > "$DONE_IDS_FILE" 2>/dev/null || true
DONE_COUNT=$(wc -l < "$DONE_IDS_FILE")
log "Resuming: $DONE_COUNT runs already completed, $((TOTAL_RUNS - DONE_COUNT)) remaining"

# ---------------------------------------------------------------------------
# 7. Dispatch function — wraps measure.py under taskset + perf stat
# ---------------------------------------------------------------------------
PERF_TMP=$(mktemp)
JSON_TMP=$(mktemp)

# Restore CPU state on exit (re-enable turbo, reset governor to powersave)
_restore_cpu() {
    [[ -f "$TURBO_PATH" ]] && echo 0 | sudo tee "$TURBO_PATH" > /dev/null 2>&1 || true
    for gov_path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -f "$gov_path" ]] && echo "powersave" | sudo tee "$gov_path" > /dev/null 2>&1 || true
    done
}
trap '_restore_cpu; rm -f "$DONE_IDS_FILE" "$PERF_TMP" "$JSON_TMP"' EXIT

run_one() {
    local lang="$1" model_short="$2" volume="$3" phase="$4" \
          rep="$5" run_order="$6" run_id="$7"

    # All language dispatch and RAPL measurement is handled by measure.py.
    # This function just pins the process to a core and wraps it in perf stat
    # to capture CPU utilisation for the full process tree (measure.py + runner).
    local cmd_args=(
        "$VENV_PY"
        "$SCRIPTS_DIR/measure.py"
        "$lang" "$model_short" "$volume" "$phase" "$rep" "$run_order" "$run_id"
    )

    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [DRY-RUN] taskset -c $CORE perf stat ... ${cmd_args[*]}"
        return 0
    fi

    # Drop OS page cache before each run
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

    local log_file="$LOGS_DIR/${run_id}.log"
    local run_stderr_tmp
    run_stderr_tmp=$(mktemp)

    # Run measure.py under taskset + perf stat.
    # perf stat follows child processes, so task-clock covers both the Python
    # wrapper and the language runner subprocess.
    # Keep per-run logs only for failed/invalid runs.
    if ! taskset -c "$CORE" \
            perf stat --field-separator ',' \
                      -e task-clock:u \
                      -o "$PERF_TMP" \
                      -- "${cmd_args[@]}" > "$JSON_TMP" 2>"$run_stderr_tmp"; then
        mv "$run_stderr_tmp" "$log_file"
        log "  WARNING: run $run_id failed (lang=$lang model=$model_short) — see $log_file"
        return 0   # continue experiment; do not abort
    fi

    # Validate JSON output before proceeding
    if ! jq -e . "$JSON_TMP" > /dev/null 2>&1; then
        mv "$run_stderr_tmp" "$log_file"
        log "  WARNING: run $run_id produced invalid JSON — see $log_file"
        return 0
    fi

    rm -f "$run_stderr_tmp"

    # Parse CPU utilisation from perf stat csv (field 6 = CPUs utilized, 0–1 scale)
    local cpu_util
    cpu_util=$(awk -F',' '/task-clock/ {printf "%.2f", $6 * 100}' "$PERF_TMP")
    [[ -z "$cpu_util" ]] && cpu_util="0.0"

    # Inject cpu_util_pct into the JSON output and append to measurements.json
    local json_out
    json_out=$(jq --argjson u "$cpu_util" '.resources.cpu_util_pct = $u' "$JSON_TMP")

    # Append to measurements.json (rebuild array each time for safety)
    local tmp_meas
    tmp_meas=$(mktemp)
    jq --argjson new_obj "$json_out" '. + [$new_obj]' "$MEAS_FILE" > "$tmp_meas"
    mv "$tmp_meas" "$MEAS_FILE"
}

# ---------------------------------------------------------------------------
# 8. Main experiment loop
# ---------------------------------------------------------------------------
log "Starting experiment on core $CORE (sleep=${SLEEP_SECS}s between runs)..."
log "============================================================"

RUN_INDEX=0
while IFS= read -r entry; do
    RUN_INDEX=$((RUN_INDEX + 1))

    run_id=$(echo "$entry"    | jq -r '.run_id')
    lang=$(echo "$entry"      | jq -r '.language')
    model_short=$(echo "$entry" | jq -r '.model_short')
    volume=$(echo "$entry"    | jq -r '.volume')
    phase=$(echo "$entry"     | jq -r '.phase')
    rep=$(echo "$entry"       | jq -r '.repetition')
    run_order=$(echo "$entry" | jq -r '.run_order')

    # Apply language filter if set
    if [[ -n "$FILTER_LANGS" ]]; then
        if ! echo "$FILTER_LANGS" | grep -qw "$lang"; then
            continue
        fi
    fi

    # Skip completed runs (resume support)
    if grep -qF "$run_id" "$DONE_IDS_FILE" 2>/dev/null; then
        continue
    fi

    log "[$run_order/$TOTAL_RUNS] lang=$lang model=$model_short vol=$volume phase=$phase rep=$rep"

    run_one "$lang" "$model_short" "$volume" "$phase" "$rep" "$run_order" "$run_id"

    if [[ "$DRY_RUN" != "true" ]]; then
        log "  -> done. Sleeping ${SLEEP_SECS}s..."
        sleep "$SLEEP_SECS"
    else
        log "  -> done."
    fi

done < <(jq -c '.[]' "$RUN_ORDER_FILE")

log "============================================================"
log "Experiment complete. Results in: $MEAS_FILE"
FINAL_COUNT=$(jq 'length' "$MEAS_FILE")
log "Total completed runs: $FINAL_COUNT / $TOTAL_RUNS"
