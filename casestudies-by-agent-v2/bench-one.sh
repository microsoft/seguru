#!/usr/bin/env bash
# Re-runs one suite and refreshes the derived results, without touching the
# other suites' numbers.
#
#   ./bench-one.sh <suite> [stock|nobc|both] [--no-plot]
#
# where <suite> is one of aes, heongpu, polybench, kernelbench, gpusorting.
#
# bench-all.sh rebuilds benchdata/all.csv wholesale from benchdata/raw/, so
# iterating on a single kernel with it costs a full re-measurement of everything
# and shifts every number in the paper. This script instead measures one suite
# and splices its rows into both raw/<variant>.csv and all.csv, replacing only
# the rows whose `suite` column matches. Keeping raw/ in step matters: a later
# bench-all.sh merge reads raw/, so a suite refreshed only in all.csv would be
# silently reverted.
set -uo pipefail

cd "$(dirname "$0")"
source ./env.sh

SUITE="${1:-}"
WHICH="${2:-both}"
PLOT=1
for a in "$@"; do [ "$a" = "--no-plot" ] && PLOT=0; done

case "$SUITE" in
    aes)         BIN=aes-bench;         FEATURES="bench" ;;
    heongpu)     BIN=heongpu-bench;     FEATURES="bench" ;;
    polybench)   BIN=polybench-bench;   FEATURES="bench" ;;
    gpusorting)  BIN=sort-bench;        FEATURES="bench" ;;
    kernelbench) BIN=kernelbench-bench; FEATURES="" ;;
    *) echo "usage: $0 <aes|heongpu|polybench|kernelbench|gpusorting> [stock|nobc|both] [--no-plot]"; exit 2 ;;
esac

OUT=benchdata
mkdir -p "$OUT/raw"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

build() {
    local targetdir="$1"; shift
    local feats=()
    [ -n "$FEATURES" ] && feats=(--features "$FEATURES")
    CARGO_TARGET_DIR="$targetdir" cargo build --release "${feats[@]}" "$@" \
        >"$TMP/build.log" 2>&1 \
        || { echo "BUILD FAILED ($targetdir $*)"; tail -20 "$TMP/build.log"; return 1; }
}

run_variant() {
    local variant="$1" targetdir="$2"
    local csv="$TMP/$variant.csv"

    echo "### building $SUITE/$variant"
    build "$targetdir" --bin "$BIN" || return 1

    echo "### running $SUITE/$variant"
    BENCH_CSV="$TMP/$variant.csv" "$targetdir/release/$BIN" \
        >"$OUT/raw/${variant}.${BIN}.txt" 2>&1
    echo "    rc=$?  rows=$( [ -f "$csv" ] && wc -l < "$csv" || echo 0 )"

    # GPUSorting also reports a second onesweep whose global scatter goes through
    # the safe atomic API; it names itself radix_sort_onesweep_safe.
    if [ "$SUITE" = "gpusorting" ]; then
        echo "### building $SUITE/$variant (safe_only)"
        CARGO_TARGET_DIR="$targetdir-safe" cargo build --release \
            --features bench,safe_only --bin sort-bench >"$TMP/build.log" 2>&1 \
            || { echo "BUILD FAILED (safe_only)"; tail -20 "$TMP/build.log"; return 1; }
        BENCH_CSV="$TMP/$variant.csv" "$targetdir-safe/release/sort-bench" \
            >"$OUT/raw/${variant}.sort-bench-safe.txt" 2>&1
        echo "    rc=$?  rows=$(wc -l < "$csv")"
    fi

    # KernelBench's CUDA mirror is a standalone nvcc binary rather than a row
    # emitted by the Rust bench. It is variant-independent and ratios() keys off
    # the stock CUDA number, so it is measured once, in the stock pass.
    if [ "$SUITE" = "kernelbench" ] && [ "$variant" = "stock" ]; then
        echo "### running $SUITE/cuda"
        if make -C kernelbench/cuda >"$TMP/build.log" 2>&1; then
            BENCH_CSV="$TMP/$variant.csv" ./kernelbench/cuda/kernelbench-cuda \
                >"$OUT/raw/stock.kernelbench-cuda.txt" 2>&1
            echo "    rc=$?  rows=$(wc -l < "$csv")"
        else
            echo "    BUILD FAILED (kernelbench-cuda)"; tail -5 "$TMP/build.log"; return 1
        fi
    fi
}

[ "$WHICH" = "stock" ] || [ "$WHICH" = "both" ] && { run_variant stock target || exit 1; }
[ "$WHICH" = "nobc" ] || [ "$WHICH" = "both" ] && { DISABLE_GPU_BOUND_CHECK=true run_variant nobc target-nobc || exit 1; }

SUITE="$SUITE" OUT="$OUT" TMP="$TMP" python3 - <<'PY'
import csv, os, sys

suite, out, tmp = os.environ['SUITE'], os.environ['OUT'], os.environ['TMP']
fresh = {}
for v in ('stock', 'nobc'):
    p = f'{tmp}/{v}.csv'
    if os.path.exists(p):
        fresh[v] = [r for r in csv.reader(open(p)) if r]

if not fresh:
    sys.exit(f'no rows measured for {suite}')

# raw/<variant>.csv has no header and no variant column.
for v, rows in fresh.items():
    p = f'{out}/raw/{v}.csv'
    old = [r for r in csv.reader(open(p))] if os.path.exists(p) else []
    kept = [r for r in old if r and r[0] != suite]
    csv.writer(open(p, 'w', newline='')).writerows(kept + rows)
    print(f'raw/{v}.csv: {len(old) - len(kept)} old {suite} rows -> {len(rows)} new')

allp = f'{out}/all.csv'
rows = list(csv.reader(open(allp)))
hdr, body = rows[0], rows[1:]
kept = [r for r in body if r[1] != suite]
new = [[v] + r for v, rs in fresh.items() for r in rs]
w = csv.writer(open(allp, 'w', newline=''))
w.writerow(hdr); w.writerows(kept + new)
print(f'all.csv: {len(body) - len(kept)} old {suite} rows -> {len(new)} new, {len(kept) + len(new)} total')
PY
[ $? -ne 0 ] && exit 1

if [ "$PLOT" = "1" ]; then
    echo "### regenerating ratios/tables/figures"
    python3.11 plot-bench.py || exit 1
    python3.11 plot-aibench-fig.py || exit 1
fi
