#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$SCRIPT_DIR/results"

echo "=== Building benchmark binary (release) ==="
cd "$REPO_ROOT" && cargo build -p kernelbench --features bench --bin bench --release 2>&1 | tail -3

echo "=== Running SeGuRu + CUDA benchmark ==="
"$REPO_ROOT/target/release/bench" > "$SCRIPT_DIR/results/seguru_cuda.csv" 2>"$SCRIPT_DIR/results/seguru_cuda.log"
echo "  $(wc -l < "$SCRIPT_DIR/results/seguru_cuda.csv") rows written to results/seguru_cuda.csv"

echo "=== Running PyTorch benchmark ==="
if command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
else
    PYTHON=python3
fi
cd /tmp && $PYTHON "$SCRIPT_DIR/bench_pytorch.py" > "$SCRIPT_DIR/results/pytorch.csv" 2>"$SCRIPT_DIR/results/pytorch.log"
echo "  $(wc -l < "$SCRIPT_DIR/results/pytorch.csv") rows written to results/pytorch.csv"

echo "=== Generating comparison report ==="
$PYTHON "$SCRIPT_DIR/compare.py"

echo ""
echo "=== Done! ==="
echo "Results in: $SCRIPT_DIR/results/"
echo "  - seguru_cuda.csv"
echo "  - pytorch.csv"
echo "  - comparison.csv"
echo "  - BENCHMARK_REPORT.md"
