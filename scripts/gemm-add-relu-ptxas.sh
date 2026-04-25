#!/usr/bin/env bash
# Compile only the standalone gemm_add_relu fixture and report ptxas regs/spills.
#
# Usage:
#   CUDA_PREFIX=/usr/local/cuda-13.2 LLVM_PREFIX=/usr/lib/llvm-20 \
#     ./scripts/gemm-add-relu-ptxas.sh --minnctapersm both

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage: scripts/gemm-add-relu-ptxas.sh [--minnctapersm 1|2|both] [--keep-tmp]
USAGE
}

minnctapersm="2"
keep_tmp=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --minnctapersm)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            minnctapersm="$2"
            shift 2
            ;;
        --minnctapersm=*)
            minnctapersm="${1#--minnctapersm=}"
            shift
            ;;
        --keep-tmp)
            keep_tmp=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

case "$minnctapersm" in
    1|2|both) ;;
    *)
        echo "ERROR: --minnctapersm must be 1, 2, or both" >&2
        exit 2
        ;;
esac

ROOT="$(git rev-parse --show-toplevel)"
CUDA_PREFIX="${CUDA_PREFIX:-/usr/local/cuda-13.2}"
LLVM_PREFIX="${LLVM_PREFIX:-/usr/lib/llvm-20}"
CRATES_DIR="$ROOT/crates"
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
    case "$CARGO_TARGET_DIR" in
        /*) TARGET_DIR="$CARGO_TARGET_DIR" ;;
        *) TARGET_DIR="$CRATES_DIR/$CARGO_TARGET_DIR" ;;
    esac
else
    TARGET_DIR="$CRATES_DIR/target"
fi
PROFILE_DIR="$TARGET_DIR/debug"
DEPS_DIR="$PROFILE_DIR/deps"
FIXTURE="$CRATES_DIR/rustc_codegen_gpu/tests/fixtures/gemm_add_relu_ptxas.rs"
KERNEL_NAME="gemm_add_relu_ptxas_kernel"

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: missing $2 at $1" >&2
        exit 1
    fi
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

export PATH="$CUDA_PREFIX/bin:$LLVM_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PREFIX/lib64:$LLVM_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$CUDA_PREFIX"
export DISABLE_GPU_BOUND_CHECK="${DISABLE_GPU_BOUND_CHECK:-true}"
export RUST_MIN_STACK="${RUST_MIN_STACK:-33554432}"

require_cmd rustc
require_cmd cargo
require_cmd ptxas
require_cmd python3
require_file "$FIXTURE" "fixture"

TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/seguru-gemm-add-relu-ptxas.XXXXXX")"
if [ "$keep_tmp" -eq 0 ]; then
    trap 'rm -rf "$TMPDIR"' EXIT
else
    echo "tmpdir=$TMPDIR" >&2
fi

run_logged() {
    local log="$1"
    shift
    if ! "$@" > "$log" 2>&1; then
        cat "$log" >&2
        exit 1
    fi
}

if ! (
    cd "$CRATES_DIR"
    cargo build -p rustc_codegen_gpu -p gpu_macros -p gpu
) > "$TMPDIR/cargo-build.log" 2>&1; then
    cat "$TMPDIR/cargo-build.log" >&2
    exit 1
fi

BACKEND="$PROFILE_DIR/librustc_codegen_gpu.so"
GPU_MACROS="$(ls -t "$DEPS_DIR"/libgpu_macros-*.so 2>/dev/null | head -n 1 || true)"
NUM_TRAITS="$(ls -t "$DEPS_DIR"/libnum_traits-*.rlib 2>/dev/null | head -n 1 || true)"

require_file "$BACKEND" "rustc_codegen_gpu backend"
require_file "$GPU_MACROS" "gpu_macros proc macro"
require_file "$NUM_TRAITS" "num_traits rlib"

GPU_OUT="$TMPDIR/gpu"
FIXTURE_OUT="$TMPDIR/fixture"
mkdir -p "$GPU_OUT" "$FIXTURE_OUT"

common_rustc_flags=(
    --edition=2021
    -C opt-level=3
    -C codegen-units=1
    --crate-type=lib
    -L "dependency=$DEPS_DIR"
    --extern "gpu_macros=$GPU_MACROS"
    --extern "num_traits=$NUM_TRAITS"
    --cfg gpu_codegen
    "-Zcodegen-backend=$BACKEND"
)

export __CODEGEN_TARGET__=GPU

run_logged "$TMPDIR/rustc-gpu.log" rustc "${common_rustc_flags[@]}" \
    --out-dir "$GPU_OUT" \
    --cfg 'feature="codegen_tests"' \
    --crate-name gpu \
    "$CRATES_DIR/gpu/src/lib.rs"

GPU_RLIB="$(find "$GPU_OUT" -maxdepth 1 -name 'libgpu*.rlib' -print | head -n 1)"
require_file "$GPU_RLIB" "compiled gpu rlib"

run_logged "$TMPDIR/rustc-fixture.log" rustc "${common_rustc_flags[@]}" \
    --emit=llvm-ir \
    --extern "gpu=$GPU_RLIB" \
    --out-dir "$FIXTURE_OUT" \
    --crate-name gemm_add_relu_ptxas \
    "$FIXTURE"

PTX="$(find "$TMPDIR" -name '*.ptx' -type f -print | while IFS= read -r ptx; do
    if grep -q "$KERNEL_NAME" "$ptx"; then
        printf '%s\n' "$ptx"
        break
    fi
done)"

require_file "$PTX" "fixture PTX containing $KERNEL_NAME"

run_ptxas() {
    local min="$1"
    local patched_ptx="$TMPDIR/gemm_add_relu_ptxas.min${min}.ptx"
    local cubin="$TMPDIR/gemm_add_relu_ptxas.min${min}.cubin"
    local log="$TMPDIR/gemm_add_relu_ptxas.min${min}.ptxas.log"

    if ! grep -q '\.minnctapersm ' "$PTX"; then
        echo "ERROR: PTX does not contain .minnctapersm" >&2
        exit 1
    fi

    sed "0,/\\.minnctapersm [0-9][0-9]*/s//.minnctapersm ${min}/" "$PTX" > "$patched_ptx"

    if ! ptxas -O3 --warn-on-spills --return-at-end --gpu-name sm_80 -v \
        -o "$cubin" "$patched_ptx" > "$log" 2>&1; then
        cat "$log" >&2
        exit 1
    fi

    python3 - "$log" "$KERNEL_NAME" "$min" <<'PY'
import re
import sys

log_path, kernel_name, minnctapersm = sys.argv[1:4]
lines = open(log_path, encoding="utf-8").read().splitlines()

for index, line in enumerate(lines):
    if "Compiling entry function" in line and kernel_name in line:
        window = "\n".join(lines[index:index + 8])
        spills = re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
            window,
        )
        regs = re.search(r"Used (\d+) registers", window)
        if spills is None or regs is None:
            print(window, file=sys.stderr)
            raise SystemExit("ERROR: failed to parse ptxas summary")
        print(
            f"minnctapersm={minnctapersm} "
            f"registers={regs.group(1)} "
            f"spill_stores={spills.group(2)} "
            f"spill_loads={spills.group(3)}"
        )
        break
else:
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(f"ERROR: ptxas entry not found for {kernel_name}")
PY
}

case "$minnctapersm" in
    both)
        run_ptxas 1
        run_ptxas 2
        ;;
    *)
        run_ptxas "$minnctapersm"
        ;;
esac
