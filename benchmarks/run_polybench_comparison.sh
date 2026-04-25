#!/usr/bin/env bash
# PolybenchGPU Performance Comparison: CUDA C++ vs SeGuRu
# GPU: NVIDIA A100 80GB PCIe (sm_80)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CUDA_DIR="$SCRIPT_DIR/cuda"
CUDA_BUILD_DIR="${POLYBENCH_CUDA_BUILD_DIR:-$SCRIPT_DIR/cuda_build}"
NVCC="${CUDA_HOME:-/usr/local/cuda-13.2}/bin/nvcc"
NVCC_FLAGS="-O3 -arch=sm_80"
LAUNCH_OVERHEAD_ITERS="${POLYBENCH_LAUNCH_OVERHEAD_ITERS:-20000}"
LAUNCH_OVERHEAD_CUDA_US="${POLYBENCH_CUDA_LAUNCH_OVERHEAD_US:-}"
LAUNCH_OVERHEAD_SEGURU_US="${POLYBENCH_SEGURU_LAUNCH_OVERHEAD_US:-}"

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

declare -A CUDA_LAUNCH_COUNTS=(
    [conv2d]=1
    [conv3d]=1
    [gemm]=1
    [twomm]=2
    [threemm]=3
    [atax]=2
    [bicg]=2
    [mvt]=2
    [gesummv]=1
    [syr2k]=1
    [syrk]=1
    [corr]=4
    [covar]=3
    [doitgen]=2
    [fdtd2d]=1500
    [gramschm]=8190
    [jacobi1d]=20000
    [jacobi2d]=40
    [lu]=4095
)

declare -A SEGURU_LAUNCH_COUNTS=(
    [conv2d]=1
    [conv3d]=1
    [gemm]=1
    [twomm]=2
    [threemm]=3
    [atax]=2
    [bicg]=2
    [mvt]=2
    [gesummv]=1
    [syr2k]=1
    [syrk]=1
    [corr]=4
    [covar]=3
    [doitgen]=2
    [fdtd2d]=1500
    [gramschm]=8192
    [jacobi1d]=20000
    [jacobi2d]=40
    [lu]=6141
)

ratio() {
    awk -v num="$1" -v den="$2" 'BEGIN { if (den <= 0) print "N/A"; else printf "%.3f", num / den }'
}

normalize_time() {
    awk -v raw="$1" -v launches="$2" -v overhead="$3" 'BEGIN {
        value = raw - launches * overhead;
        if (value < 0.001) value = 0.001;
        printf "%.3f", value;
    }'
}

extract_us() {
    grep -oP '[\d.]+(?= us/iter)' | head -1 || true
}

require_launch_overhead() {
    local side="$1"
    local value="$2"
    local source="$3"

    if [ -z "$value" ]; then
        echo "ERROR: failed to extract $side launch overhead from: $source" >&2
        exit 1
    fi
}

echo "================================================================"
echo "PolybenchGPU Performance Comparison: CUDA C++ vs SeGuRu"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "NVCC: $NVCC"
echo "================================================================"
echo ""

# ---------- Phase 1: Compile CUDA benchmarks ----------
echo "--- Phase 1: Compiling CUDA benchmarks ---"
mkdir -p "$CUDA_BUILD_DIR"
CUDA_RESULTS_FILE="$SCRIPT_DIR/cuda_results.txt"
: > "$CUDA_RESULTS_FILE"

for bench in "${BENCHMARKS[@]}"; do
    src="$CUDA_DIR/bench_${bench}.cu"
    bin="$CUDA_BUILD_DIR/bench_${bench}"
    if [ ! -f "$src" ]; then
        echo "SKIP $bench (no $src)"
        continue
    fi
    echo "  Compiling $(basename "$src") ..."
    $NVCC $NVCC_FLAGS -o "$bin" "$src" 2>&1
done

echo "  Compiling bench_empty_launch.cu ..."
$NVCC $NVCC_FLAGS -o "$CUDA_BUILD_DIR/bench_empty_launch" "$CUDA_DIR/bench_empty_launch.cu" 2>&1
echo ""

# ---------- Phase 2: Run CUDA benchmarks ----------
echo "--- Phase 2: Running CUDA benchmarks ---"
for bench in "${BENCHMARKS[@]}"; do
    bin="$CUDA_BUILD_DIR/bench_${bench}"
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

# ---------- Phase 4: Measure launch overhead ----------
echo "--- Phase 4: Measuring launch overhead ---"
if [ -z "$LAUNCH_OVERHEAD_CUDA_US" ]; then
    cuda_launch_result=$("$CUDA_BUILD_DIR/bench_empty_launch" "$LAUNCH_OVERHEAD_ITERS" 2>/dev/null | head -1)
    LAUNCH_OVERHEAD_CUDA_US=$(printf '%s\n' "$cuda_launch_result" | extract_us)
else
    cuda_launch_result="empty_launch CUDA: ${LAUNCH_OVERHEAD_CUDA_US} us/iter (${LAUNCH_OVERHEAD_ITERS} iters, env override)"
fi
require_launch_overhead "CUDA" "$LAUNCH_OVERHEAD_CUDA_US" "$cuda_launch_result"
if [ -z "$LAUNCH_OVERHEAD_SEGURU_US" ]; then
    seguru_launch_result=$(cargo run --release -p bench-polybench -- --launch-overhead "$LAUNCH_OVERHEAD_ITERS" 2>/dev/null | head -1)
    LAUNCH_OVERHEAD_SEGURU_US=$(printf '%s\n' "$seguru_launch_result" | extract_us)
else
    seguru_launch_result="empty_launch SeGuRu: ${LAUNCH_OVERHEAD_SEGURU_US} us/iter (${LAUNCH_OVERHEAD_ITERS} iters, env override)"
fi
require_launch_overhead "SeGuRu" "$LAUNCH_OVERHEAD_SEGURU_US" "$seguru_launch_result"
echo "  $cuda_launch_result"
echo "  $seguru_launch_result"
echo ""

# ---------- Phase 5: Run SeGuRu benchmark ----------
echo "--- Phase 5: Running SeGuRu bench-polybench ---"
SEGURU_RESULTS_FILE="$SCRIPT_DIR/seguru_results.txt"
cargo run --release -p bench-polybench 2>/dev/null | tee "$SEGURU_RESULTS_FILE"
echo ""

# ---------- Phase 6: Print comparison tables ----------
echo ""
echo "================================================================"
echo "                    RAW COMPARISON TABLE"
echo "================================================================"
printf "%-12s %15s %15s %10s\n" "Benchmark" "CUDA (us)" "SeGuRu (us)" "Ratio"
printf "%-12s %15s %15s %10s\n" "--------" "---------" "----------" "-----"

for bench in "${BENCHMARKS[@]}"; do
    cuda_us=$(grep "^${bench} CUDA:" "$CUDA_RESULTS_FILE" 2>/dev/null | grep -oP '[\d.]+(?= us/iter)' || echo "N/A")
    seguru_us=$(grep "^${bench} SeGuRu:" "$SEGURU_RESULTS_FILE" 2>/dev/null | grep -oP '[\d.]+(?= us/iter)' || echo "N/A")

    if [ "$cuda_us" != "N/A" ] && [ "$seguru_us" != "N/A" ]; then
        raw_ratio=$(ratio "$seguru_us" "$cuda_us")
        printf "%-12s %15s %15s %9sx\n" "$bench" "$cuda_us" "$seguru_us" "$raw_ratio"
    else
        printf "%-12s %15s %15s %10s\n" "$bench" "$cuda_us" "$seguru_us" "N/A"
    fi
done

echo "================================================================"
echo "Ratio = SeGuRu / CUDA (lower is better for SeGuRu)"
echo "================================================================"

echo ""
echo "================================================================"
echo "              LAUNCH-NORMALIZED COMPARISON TABLE"
echo "================================================================"
echo "CUDA launch overhead:   ${LAUNCH_OVERHEAD_CUDA_US} us/launch"
echo "SeGuRu launch overhead: ${LAUNCH_OVERHEAD_SEGURU_US} us/launch"
echo "Normalized time = raw time - side-specific launch count * side-specific launch overhead"
echo "Normalized times are floored at 0.001 us when launch overhead dominates."
echo "Ratios below 1.0x in the launch-normalized table can indicate launch overhead"
echo "dominates the measured runtime; use the raw table as the authoritative"
echo "wall-clock result for those workloads."
printf "%-12s %13s %15s %15s %15s %15s\n" "Benchmark" "CUDA launches" "SeGuRu launches" "CUDA norm(us)" "SeGuRu norm(us)" "Norm Ratio"
printf "%-12s %13s %15s %15s %15s %15s\n" "--------" "-------------" "---------------" "-------------" "---------------" "----------"

for bench in "${BENCHMARKS[@]}"; do
    cuda_us=$(grep "^${bench} CUDA:" "$CUDA_RESULTS_FILE" 2>/dev/null | grep -oP '[\d.]+(?= us/iter)' || echo "N/A")
    seguru_us=$(grep "^${bench} SeGuRu:" "$SEGURU_RESULTS_FILE" 2>/dev/null | grep -oP '[\d.]+(?= us/iter)' || echo "N/A")
    cuda_launch_count="${CUDA_LAUNCH_COUNTS[$bench]}"
    seguru_launch_count="${SEGURU_LAUNCH_COUNTS[$bench]}"

    if [ "$cuda_us" != "N/A" ] && [ "$seguru_us" != "N/A" ]; then
        cuda_norm=$(normalize_time "$cuda_us" "$cuda_launch_count" "$LAUNCH_OVERHEAD_CUDA_US")
        seguru_norm=$(normalize_time "$seguru_us" "$seguru_launch_count" "$LAUNCH_OVERHEAD_SEGURU_US")
        norm_ratio=$(ratio "$seguru_norm" "$cuda_norm")
        printf "%-12s %13s %15s %15s %15s %14sx\n" "$bench" "$cuda_launch_count" "$seguru_launch_count" "$cuda_norm" "$seguru_norm" "$norm_ratio"
    else
        printf "%-12s %13s %15s %15s %15s %15s\n" "$bench" "$cuda_launch_count" "$seguru_launch_count" "$cuda_us" "$seguru_us" "N/A"
    fi
done

echo "================================================================"
echo "Normalized ratio = launch-normalized SeGuRu / launch-normalized CUDA"
echo "================================================================"
