#!/usr/bin/env bash
# PolybenchGPU Performance Comparison: CUDA C++ vs SeGuRu
# GPU: NVIDIA A100 80GB PCIe (sm_80)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CUDA_DIR="$SCRIPT_DIR/cuda"
NVCC="${CUDA_HOME:-/usr/local/cuda-13.2}/bin/nvcc"
NVCC_FLAGS="-O3 -arch=sm_80"

export PATH="${CUDA_HOME:-/usr/local/cuda-13.2}/bin:/usr/lib/llvm-20/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda-13.2}/lib64:/usr/lib/llvm-20/lib:${LD_LIBRARY_PATH:-}"

# All benchmarks (19 that exist in both CUDA and SeGuRu)
BENCHMARKS=(
    conv2d conv3d gemm twomm threemm
    atax bicg mvt gesummv
    syr2k syrk
    corr covar doitgen
    fdtd2d gramschm
    jacobi1d jacobi2d lu
)

echo "================================================================"
echo "PolybenchGPU Performance Comparison: CUDA C++ vs SeGuRu"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "NVCC: $NVCC"
echo "================================================================"
echo ""

# ---------- Phase 1: Compile CUDA benchmarks ----------
echo "--- Phase 1: Compiling CUDA benchmarks ---"
cd "$CUDA_DIR"
CUDA_RESULTS_FILE="$SCRIPT_DIR/cuda_results.txt"
: > "$CUDA_RESULTS_FILE"

for bench in "${BENCHMARKS[@]}"; do
    src="bench_${bench}.cu"
    bin="bench_${bench}"
    if [ ! -f "$src" ]; then
        echo "SKIP $bench (no $src)"
        continue
    fi
    echo "  Compiling $src ..."
    $NVCC $NVCC_FLAGS -o "$bin" "$src" 2>&1
done
echo ""

# ---------- Phase 2: Run CUDA benchmarks ----------
echo "--- Phase 2: Running CUDA benchmarks ---"
for bench in "${BENCHMARKS[@]}"; do
    bin="$CUDA_DIR/bench_${bench}"
    if [ ! -x "$bin" ]; then
        echo "  SKIP $bench (not compiled)"
        echo "$bench CUDA: SKIP" >> "$CUDA_RESULTS_FILE"
        continue
    fi
    result=$("$bin" 2>/dev/null | head -1)
    echo "  $result"
    echo "$result" >> "$CUDA_RESULTS_FILE"
done
echo ""

# ---------- Phase 3: Build SeGuRu benchmark ----------
echo "--- Phase 3: Building SeGuRu bench-polybench ---"
cd "$REPO_ROOT/examples"
cargo build --release -p bench-polybench 2>&1 | tail -3
echo ""

# ---------- Phase 4: Run SeGuRu benchmark ----------
echo "--- Phase 4: Running SeGuRu bench-polybench ---"
SEGURU_RESULTS_FILE="$SCRIPT_DIR/seguru_results.txt"
cargo run --release -p bench-polybench 2>/dev/null | tee "$SEGURU_RESULTS_FILE"
echo ""

# ---------- Phase 5: Print comparison table ----------
echo ""
echo "================================================================"
echo "                    COMPARISON TABLE"
echo "================================================================"
printf "%-12s %15s %15s %10s\n" "Benchmark" "CUDA (us)" "SeGuRu (us)" "Ratio"
printf "%-12s %15s %15s %10s\n" "--------" "---------" "----------" "-----"

for bench in "${BENCHMARKS[@]}"; do
    cuda_us=$(grep "^${bench} CUDA:" "$CUDA_RESULTS_FILE" 2>/dev/null | grep -oP '[\d.]+(?= us/iter)' || echo "N/A")
    seguru_us=$(grep "^${bench} SeGuRu:" "$SEGURU_RESULTS_FILE" 2>/dev/null | grep -oP '[\d.]+(?= us/iter)' || echo "N/A")

    if [ "$cuda_us" != "N/A" ] && [ "$seguru_us" != "N/A" ]; then
        ratio=$(echo "scale=3; $seguru_us / $cuda_us" | bc 2>/dev/null || echo "N/A")
        printf "%-12s %15s %15s %9sx\n" "$bench" "$cuda_us" "$seguru_us" "$ratio"
    else
        printf "%-12s %15s %15s %10s\n" "$bench" "$cuda_us" "$seguru_us" "N/A"
    fi
done

echo "================================================================"
echo "Ratio = SeGuRu / CUDA (lower is better for SeGuRu)"
echo "================================================================"
