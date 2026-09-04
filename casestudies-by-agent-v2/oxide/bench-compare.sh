#!/usr/bin/env bash
# Run both sides of the SeGuRu / cuda-oxide comparison and write the raw CSVs
# plus a merged comparison table into `results/`.
#
#   ./bench-compare.sh
#
# The SeGuRu numbers come from the existing benchmark binaries in the parent
# directory; the cuda-oxide numbers from `bench-oxide.sh`. Both sides use the
# same input generator, the same problem sizes, the same iteration counts and
# the same kernel-only timing methodology, so the rows line up directly.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASESTUDIES="$(dirname "$HERE")"
OUT="$HERE/results"
mkdir -p "$OUT"

echo "==> cuda-oxide"
"$HERE/bench-oxide.sh" > "$OUT/oxide.csv"

echo "==> SeGuRu (polybench: gemm, atax)"
(
    cd "$CASESTUDIES"
    # shellcheck source=/dev/null
    source ./env.sh
    cargo run --release -p polybench-gpu --features bench --bin polybench-bench gemm
    cargo run --release -p polybench-gpu --features bench --bin polybench-bench atax
) > "$OUT/seguru-polybench.txt"

echo "==> SeGuRu (gpusorting: radix_upsweep)"
(
    cd "$CASESTUDIES"
    # shellcheck source=/dev/null
    source ./env.sh
    cargo run --release -p gpusorting-gpu --bin upsweep-bench
) | sed -n '/^kernel,/,$p' > "$OUT/seguru-upsweep.csv"

echo "==> metrics"
"$HERE/metrics.py" > "$OUT/metrics.md"

echo
echo "wrote:"
ls -1 "$OUT"
