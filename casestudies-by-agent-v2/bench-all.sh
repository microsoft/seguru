#!/usr/bin/env bash
# Runs every case-study benchmark and collects one machine-readable CSV.
#
# Each suite's bench binary appends rows to $BENCH_CSV when that variable is set
# (schema: suite,workload,parameter,implementation,metric,value,units). This
# script runs the whole set twice:
#
#   stock  -- SeGuRu as shipped, bounds checks emitted
#   nobc   -- rebuilt with DISABLE_GPU_BOUND_CHECK=true, which makes
#             rustc_codegen_gpu skip the per-access bounds check
#
# and merges them into benchdata/all.csv with a leading `variant` column, so the
# cost of the safety check is a column rather than a separate experiment.
#
# The two variants build into separate target directories: DISABLE_GPU_BOUND_CHECK
# is read by the backend at codegen time and cargo cannot see it, so sharing a
# target dir would silently reuse stale artifacts.
set -uo pipefail

cd "$(dirname "$0")"
source ./env.sh

OUT=benchdata
mkdir -p "$OUT/raw"

run_variant() {
    local variant="$1" targetdir="$2"
    local csv="$OUT/raw/${variant}.csv"
    rm -f "$csv"

    echo "### building $variant (CARGO_TARGET_DIR=$targetdir)"
    CARGO_TARGET_DIR="$targetdir" cargo build --release --features bench \
        --bin aes-bench --bin heongpu-bench --bin polybench-bench --bin sort-bench \
        >"$OUT/raw/${variant}.build.log" 2>&1 || { echo "BUILD FAILED"; tail -20 "$OUT/raw/${variant}.build.log"; return 1; }
    CARGO_TARGET_DIR="$targetdir" cargo build --release --bin kernelbench-bench \
        >>"$OUT/raw/${variant}.build.log" 2>&1 || { echo "BUILD FAILED (kernelbench)"; return 1; }

    for b in aes-bench heongpu-bench polybench-bench kernelbench-bench sort-bench; do
        echo "### $variant: $b"
        BENCH_CSV="$PWD/$csv" "$targetdir/release/$b" \
            >"$OUT/raw/${variant}.${b}.txt" 2>&1
        local rc=$?
        echo "    rc=$rc  rows=$( [ -f "$csv" ] && wc -l < "$csv" || echo 0 )"
    done
}

# The no-bounds-check build is the one that needs the env var at codegen time.
if [ "${1:-both}" = "stock" ] || [ "${1:-both}" = "both" ]; then
    run_variant stock target
fi
if [ "${1:-both}" = "nobc" ] || [ "${1:-both}" = "both" ]; then
    DISABLE_GPU_BOUND_CHECK=true run_variant nobc target-nobc
fi

# Merge, prefixing each row with its variant.
{
    echo "variant,suite,workload,parameter,implementation,metric,value,units"
    for v in stock nobc; do
        [ -f "$OUT/raw/$v.csv" ] && sed "s/^/$v,/" "$OUT/raw/$v.csv"
    done
} > "$OUT/all.csv"

echo "### merged $(( $(wc -l < "$OUT/all.csv") - 1 )) rows into $OUT/all.csv"
