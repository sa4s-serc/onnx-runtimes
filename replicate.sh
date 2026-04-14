#!/usr/bin/env bash
# =============================================================================
# replicate.sh — Full replication from scratch.
#
# Performs every step needed to reproduce the experiment: Python venv + deps,
# model downloads, dataset preparation, building all language binaries, and
# running measure.sh. Every step is mandatory — this script fails fast and
# loudly on any error so that no incomplete state goes undetected.
#
# Usage:
#   bash replicate.sh
#
# Prerequisites (install before running):
#   sudo apt install build-essential default-jdk maven ruby perf jq util-linux
#   # Rust (optional — skipped with a hard error if missing):
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#   # .NET SDK (optional — skipped with a hard error if missing):
#   See https://dot.net/
# =============================================================================

set -euo pipefail

# Ensure rustup-managed cargo is on PATH if installed
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/.venv"
VENV_PY="$VENV/bin/python"
VENV_PIP="$VENV/bin/pip"
SCRIPTS="$REPO_ROOT/scripts"
ORT_DYLIB="$REPO_ROOT/lib/onnxruntime/lib/libonnxruntime.so"

log() { echo ""; echo "==> $*"; }
ok()  { echo "    [OK] $*"; }
die() { echo ""; echo "[FATAL] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Python virtual environment
# ---------------------------------------------------------------------------
log "Step 1: Python virtual environment"
if [[ ! -f "$VENV_PY" ]]; then
    python3 -m venv "$VENV"
fi
ok "venv at $VENV"

# ---------------------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------------------
log "Step 2: Python dependencies"
"$VENV_PIP" install --quiet --upgrade pip
"$VENV_PIP" install --quiet \
    "onnxruntime==1.21.0" \
    "numpy" \
    "datasets" \
    "transformers" \
    "huggingface_hub" \
    "pillow"
ok "Python packages installed"

# ---------------------------------------------------------------------------
# 3. Download ONNX models
# ---------------------------------------------------------------------------
log "Step 3: Download ONNX models"
"$VENV_PY" "$SCRIPTS/download_models.py"
ok "Models in models/"

# ---------------------------------------------------------------------------
# 4. Prepare input tensors
# ---------------------------------------------------------------------------
log "Step 4: Prepare input tensors"
"$VENV_PY" "$SCRIPTS/prepare_inputs.py"
ok "Inputs in inputs/"

# ---------------------------------------------------------------------------
# 5a. Build C binary
# ---------------------------------------------------------------------------
log "Step 5a: C (gcc)"
command -v gcc &>/dev/null || die "gcc not found. Install: sudo apt install build-essential"
make -C "$SCRIPTS/inference/c" -j"$(nproc)"
ok "C binary: scripts/inference/c/infer"

# ---------------------------------------------------------------------------
# 5b. Build C++ binary
# ---------------------------------------------------------------------------
log "Step 5b: C++ (g++)"
command -v g++ &>/dev/null || die "g++ not found. Install: sudo apt install build-essential"
make -C "$SCRIPTS/inference/cpp" -j"$(nproc)"
ok "C++ binary: scripts/inference/cpp/infer"

# ---------------------------------------------------------------------------
# 5c. Build Java fat JAR (Maven)
# ---------------------------------------------------------------------------
log "Step 5c: Java (Maven)"
command -v java &>/dev/null || die "java not found. Install: sudo apt install default-jdk"
command -v mvn  &>/dev/null || die "mvn not found. Install: sudo apt install maven"
mvn -q -f "$SCRIPTS/inference/java/pom.xml" clean package -DskipTests
ok "Java JAR: scripts/inference/java/target/onnx-inference-1.0-SNAPSHOT.jar"

# ---------------------------------------------------------------------------
# 5d. Build C# binary (.NET)
# ---------------------------------------------------------------------------
log "Step 5d: C# (.NET)"
command -v dotnet &>/dev/null || die "dotnet not found. See https://dot.net/"
dotnet build "$SCRIPTS/inference/csharp/OnnxInference" \
    -c Release --nologo -v quiet
ok "C# binary built"

# ---------------------------------------------------------------------------
# 5e. Build Rust binary (cargo)
# ---------------------------------------------------------------------------
log "Step 5e: Rust (cargo)"
command -v cargo &>/dev/null || die "cargo not found. Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
ORT_DYLIB_PATH="$ORT_DYLIB" \
cargo build --release \
    --manifest-path "$SCRIPTS/inference/rust/Cargo.toml"
ok "Rust binary: scripts/inference/rust/target/release/onnx_inference"

# ---------------------------------------------------------------------------
# 5f. Install Ruby gem (onnxruntime)
# ---------------------------------------------------------------------------
log "Step 5f: Ruby (onnxruntime gem)"
command -v ruby &>/dev/null || die "ruby not found. Install: sudo apt install ruby"
# Install to user gem dir (no sudo required)
gem install --user-install onnxruntime --quiet 2>&1 | tail -3
# Verify the gem loaded correctly with our local ORT library
ORT_DYLIB_PATH="$ORT_DYLIB" ruby -r onnxruntime -e \
    "OnnxRuntime.ffi_lib = ENV['ORT_DYLIB_PATH']; puts 'Ruby onnxruntime OK'"
ok "Ruby onnxruntime gem installed"

# ---------------------------------------------------------------------------
# 6. Run measurements
# ---------------------------------------------------------------------------
log "Step 6: Run full experiment (measure.sh)"
bash "$REPO_ROOT/measure.sh"

log "Done"
echo ""
echo "Replication complete. Results: $REPO_ROOT/data/measurements.json"
