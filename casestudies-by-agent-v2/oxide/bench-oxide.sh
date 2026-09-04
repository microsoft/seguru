#!/usr/bin/env bash
# Run every cuda-oxide benchmark in this directory and print one CSV.
#
#   ./bench-oxide.sh            # all of them
#   ./bench-oxide.sh gemm atax  # a subset, matched against the directory names
#
# Each benchmark prints its own `kernel,impl,size,...` CSV; this script keeps
# the first header and concatenates the rows.

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./env.sh

ALL=(polybench-gemm polybench-atax sort-upsweep)

if [ $# -gt 0 ]; then
    SELECTED=()
    for want in "$@"; do
        for bench in "${ALL[@]}"; do
            case "$bench" in *"$want"*) SELECTED+=("$bench") ;; esac
        done
    done
else
    SELECTED=("${ALL[@]}")
fi

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "no benchmark matched: $*" >&2
    echo "available: ${ALL[*]}" >&2
    exit 1
fi

header_printed=0
for bench in "${SELECTED[@]}"; do
    out=$(cd "$bench" && cargo oxide run 2>/dev/null | sed -n '/^kernel,/,$p')
    if [ -z "$out" ]; then
        echo "$bench: produced no CSV; run 'cd $bench && cargo oxide run' to see why" >&2
        exit 1
    fi
    if [ "$header_printed" -eq 0 ]; then
        echo "$out"
        header_printed=1
    else
        echo "$out" | tail -n +2
    fi
done
