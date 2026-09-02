#!/usr/bin/env bash
# One command for the whole benchmark pipeline.
#
#   ./run-bench.sh          # both variants, then regenerate tables and plots
#   ./run-bench.sh stock    # SeGuRu as shipped only (skips the nobc rebuild)
#   ./run-bench.sh nobc     # bounds-checks-disabled build only
#
# Each suite's bench binary is built with `--features bench`, which makes its
# build.rs compile the CUDA C++ baseline with nvcc, so one run times SeGuRu and
# CUDA together and writes both to the same CSV. There is no separate CUDA step.
#
# Outputs:
#   benchdata/all.csv                     every measurement, one row per run
#   benchdata/RESULTS.md                  human-readable summary
#   benchdata/bench_{box,detail,times}.png
#   ../seguru-paper/src/tab_appendix_polybench.tex
#   ../seguru-paper/src/tab_appendix_kernelbench.tex
set -uo pipefail
cd "$(dirname "$0")"

./bench-all.sh "${1:-both}" || { echo "benchmarks failed; not regenerating tables"; exit 1; }
python3.11 plot-bench.py || { echo "table generation failed"; exit 1; }

echo
echo "### done. CSV: benchdata/all.csv"
