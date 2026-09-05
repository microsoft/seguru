#!/usr/bin/env python3
"""Tidy benchdata/all.csv into the single CSV the paper's aibench table and
figure both read at LaTeX compile time.

Writes seguru-paper/src/data/aibench.csv. Both
``seguru-paper/src/data/aibench.tex`` (Table: every kernel vs its CUDA mirror)
and ``seguru-paper/src/figs/aibench.tex`` (Figure: per-suite box plot of the
ratios) typeset themselves from that one file, so they cannot drift apart and
the paper carries no hand-transcribed numbers.

Only what those two consumers need is exported: the bounds-checked ("stock")
variant, the suites that have a CUDA mirror, and one row per measurement. The
columns that LaTeX cannot reasonably derive itself -- the display labels, the
size class, the box-plot x offsets -- are precomputed here.

Re-run with: python3.11 export-aibench-csv.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_IN = HERE / "benchdata" / "all.csv"
OUT = HERE.parent / "seguru-paper" / "src" / "data" / "aibench.csv"

SUITE_ORDER = ["aes", "heongpu", "polybench", "kernelbench", "gpusorting"]
GPUSORT_E2E = {
    "radix_sort_onesweep",
    "radix_sort_onesweep_safe",
    "radix_sort_reduce_then_scan",
}
SUITE_TITLE = {"aes": "AES", "heongpu": "HEonGPU",
               "polybench": "PolyBench", "kernelbench": "KernelBench",
               "gpusorting": "GPUSorting"}

# The onesweep port is measured twice: once with its global scatter written
# through a single `unsafe` MapExplicit, once with that scatter on the safe
# atomic API. The table shows them as a pair so it prices that one line; the
# figure keeps only the safe build, so the box is not double-counted and every
# point in it comes from safe Rust.
DISPLAY = {
    "radix_sort_onesweep": r"onesweep (1 \code{unsafe})",
    "radix_sort_onesweep_safe": r"onesweep (safe, atomic)",
    "radix_sort_reduce_then_scan": r"reduce\_then\_scan",
    "radix_upsweep": r"upsweep (kernel)",
    "radix_scan": r"scan (kernel)",
    "radix_downsweep": r"downsweep (kernel)",
    "onesweep_global_histogram": r"global\_histogram (kernel)",
    "onesweep_scan": r"onesweep scan (kernel)",
    "onesweep_digit_binning": r"digit\_binning (kernel, 1 \code{unsafe})",
    "onesweep_digit_binning_safe": r"digit\_binning (kernel, safe, atomic)",
}
FIG_EXCLUDE = {"radix_sort_onesweep", "onesweep_digit_binning", "onesweep_scan"}
# Sort keys that put the safe row above the one that uses `unsafe`.
WORKLOAD_SORT = {
    "radix_sort_onesweep_safe": "radix_sort_onesweep_a",
    "radix_sort_onesweep": "radix_sort_onesweep_b",
    "onesweep_digit_binning_safe": "onesweep_digit_binning_a",
    "onesweep_digit_binning": "onesweep_digit_binning_b",
}

SIZE_CLASSES = ["small", "medium", "large"]
# Box plot: the three size classes are drawn at a fixed offset from the box.
SIZE_DX = {cls: -0.19 + 0.19 * i for i, cls in enumerate(SIZE_CLASSES)}


def fmt_size(p):
    """Size tokens like 4096^3, 8192^2xt10 or ring=65536 into math mode."""
    p = p.replace("=", r"{=}").replace("xt", r", $t{=}$")
    out, i = [], 0
    while i < len(p):
        if p[i] == "^":
            out.append("$^{" + p[i + 1] + "}$")
            i += 2
        else:
            out.append(p[i])
            i += 1
    return "".join(out)


SEP = ";"


def brace(field):
    """Protect a cell that contains the separator.

    Cells legitimately contain commas -- a size label like ``2048$^{2}$,
    $t{=}$10``, or the ``\\,`` thin space in ``0.3\\,ms`` -- so the file is
    semicolon-separated. pgfplotstable does not understand RFC-4180 quoting,
    and its brace handling is unreliable here, so the separator must simply
    never appear inside a cell.
    """
    return "{" + field + "}" if SEP in field else field


def load(src):
    """CUDA and SeGuRu times per (suite, workload, parameter), bounds-checked."""
    times = defaultdict(dict)
    units = {}
    for r in csv.DictReader(src.open()):
        if r["variant"] != "stock" or r["metric"] != "time":
            continue
        if r["suite"] not in SUITE_ORDER:
            continue
        if r["suite"] == "gpusorting" and r["workload"] not in GPUSORT_E2E:
            continue
        k = (r["suite"], r["workload"], r["parameter"])
        times[k][r["implementation"]] = float(r["value"])
        units.setdefault(k, r["units"])
    return {k: v for k, v in times.items()
            if "cuda" in v and "seguru" in v}, units


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", type=Path, default=CSV_IN,
                    help=f"benchmark CSV to tidy (default: {CSV_IN})")
    ap.add_argument("-o", "--output", type=Path, default=OUT,
                    help=f"where to write the tidied CSV (default: {OUT})")
    args = ap.parse_args()

    times, units = load(args.input)

    # Within a workload, sizes are ranked by CUDA runtime rather than by the
    # parameter string, which is not comparable across suites ("1GiB" vs
    # "ring=65536" vs "256Mi").
    per_workload = defaultdict(list)
    for (suite, wl, param), v in times.items():
        per_workload[(suite, wl)].append((v["cuda"], param))
    for v in per_workload.values():
        v.sort()

    size_class = {}
    for (suite, wl), sizes in per_workload.items():
        n = len(sizes)
        for i, (_cuda, param) in enumerate(sizes):
            size_class[(suite, wl, param)] = (
                "small" if i == 0 else "large" if i == n - 1 else "medium")

    rows = []
    for (suite, wl, param), v in times.items():
        ratio = v["seguru"] / v["cuda"]
        cuda, unit = v["cuda"], units[(suite, wl, param)]
        cls = size_class[(suite, wl, param)]
        rows.append({
            "suite": suite,
            "suitetitle": SUITE_TITLE[suite],
            "suiteidx": SUITE_ORDER.index(suite) + 1,
            "kernel": wl,
            "size": fmt_size(param),
            "cuda": f"{cuda:.1f}",
            "ratio": f"{ratio:.6f}",
            "sizeclass": cls,
            "x": f"{SUITE_ORDER.index(suite) + 1 + SIZE_DX[cls]:.3f}",
            "infig": int(wl not in FIG_EXCLUDE),
            "_sort": (SUITE_ORDER.index(suite), WORKLOAD_SORT.get(wl, wl), cuda),
        })

    rows.sort(key=lambda r: r["_sort"])
    for r in rows:
        del r["_sort"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    with args.output.open("w") as f:
        f.write("# GENERATED by casestudies-by-agent-v2/export-aibench-csv.py"
                f" from {args.input.name}.\n")
        f.write("# Do not edit by hand: re-run ./bench-all.sh &&"
                " python3.11 export-aibench-csv.py.\n")
        f.write(SEP.join(cols) + "\n")
        for r in rows:
            f.write(SEP.join(brace(str(r[c])) for c in cols) + "\n")

    ratios = [float(r["ratio"]) for r in rows]
    print(f"wrote {args.output} ({len(rows)} rows,"
          f" ratio {min(ratios):.3f}..{max(ratios):.3f})")


if __name__ == "__main__":
    main()
