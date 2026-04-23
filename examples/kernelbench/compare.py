#!/usr/bin/env python3.11
"""Merge SeGuRu/CUDA and PyTorch benchmark results into comparison report."""

import csv
import sys
import os
from collections import defaultdict

def read_csv(path):
    """Read CSV into list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    seguru_csv = os.path.join(results_dir, "seguru_cuda.csv")
    pytorch_csv = os.path.join(results_dir, "pytorch.csv")
    comparison_csv = os.path.join(results_dir, "comparison.csv")
    report_md = os.path.join(results_dir, "BENCHMARK_REPORT.md")

    if not os.path.exists(seguru_csv):
        print(f"ERROR: {seguru_csv} not found. Run bench binary first.", file=sys.stderr)
        sys.exit(1)

    seguru_rows = read_csv(seguru_csv)

    # Build lookup: (kernel, size_label) -> pytorch_us
    pytorch_lookup = {}
    if os.path.exists(pytorch_csv):
        for row in read_csv(pytorch_csv):
            key = (row["kernel"], row["size_label"])
            pytorch_lookup[key] = float(row["pytorch_us"])
    else:
        print("WARNING: pytorch.csv not found, skipping PyTorch comparison.", file=sys.stderr)

    # Merge
    merged = []
    for row in seguru_rows:
        key = (row["kernel"], row["size_label"])
        seguru_us = float(row["seguru_us"])
        cuda_us = float(row["cuda_us"])
        pytorch_us = pytorch_lookup.get(key)
        ratio_sc = seguru_us / cuda_us if cuda_us > 0 else float("inf")
        ratio_sp = seguru_us / pytorch_us if pytorch_us and pytorch_us > 0 else None
        ratio_cp = cuda_us / pytorch_us if pytorch_us and pytorch_us > 0 else None
        merged.append({
            "kernel": row["kernel"],
            "category": row["category"],
            "size_label": row["size_label"],
            "n_elements": row["n_elements"],
            "seguru_us": seguru_us,
            "cuda_us": cuda_us,
            "pytorch_us": pytorch_us,
            "seguru_vs_cuda": ratio_sc,
            "seguru_vs_pytorch": ratio_sp,
            "cuda_vs_pytorch": ratio_cp,
        })

    # Write comparison CSV
    with open(comparison_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "category", "size_label", "n_elements",
                         "seguru_us", "cuda_us", "pytorch_us",
                         "seguru_vs_cuda", "seguru_vs_pytorch", "cuda_vs_pytorch"])
        for m in merged:
            writer.writerow([
                m["kernel"], m["category"], m["size_label"], m["n_elements"],
                f"{m['seguru_us']:.2f}", f"{m['cuda_us']:.2f}",
                f"{m['pytorch_us']:.2f}" if m["pytorch_us"] else "N/A",
                f"{m['seguru_vs_cuda']:.4f}",
                f"{m['seguru_vs_pytorch']:.4f}" if m["seguru_vs_pytorch"] else "N/A",
                f"{m['cuda_vs_pytorch']:.4f}" if m["cuda_vs_pytorch"] else "N/A",
            ])
    print(f"Wrote {comparison_csv}", file=sys.stderr)

    # Group by category
    by_cat = defaultdict(list)
    for m in merged:
        by_cat[m["category"]].append(m)

    # Group by size
    by_size = defaultdict(list)
    for m in merged:
        by_size[m["size_label"]].append(m)

    # Compute category averages
    cat_avgs = {}
    for cat, rows in sorted(by_cat.items()):
        sc_ratios = [r["seguru_vs_cuda"] for r in rows]
        cat_avgs[cat] = sum(sc_ratios) / len(sc_ratios)

    # Overall average
    all_sc = [m["seguru_vs_cuda"] for m in merged]
    overall_sc = sum(all_sc) / len(all_sc)

    all_sp = [m["seguru_vs_pytorch"] for m in merged if m["seguru_vs_pytorch"]]
    overall_sp = sum(all_sp) / len(all_sp) if all_sp else None

    # Size breakdown
    size_avgs = {}
    for sz, rows in by_size.items():
        sc = [r["seguru_vs_cuda"] for r in rows]
        size_avgs[sz] = sum(sc) / len(sc)

    # Best/worst
    best = min(merged, key=lambda m: m["seguru_vs_cuda"])
    worst = max(merged, key=lambda m: m["seguru_vs_cuda"])

    # ===== Markdown Report =====
    lines = []
    lines.append("# KernelBench Level 1 Performance Report\n")
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total kernels benchmarked | {len(set(m['kernel'] for m in merged))} |")
    lines.append(f"| Total measurements | {len(merged)} |")
    lines.append(f"| SeGuRu avg overhead vs CUDA | **{overall_sc:.2f}×** |")
    if overall_sp:
        lines.append(f"| SeGuRu avg overhead vs PyTorch | **{overall_sp:.2f}×** |")
    lines.append(f"| Best (closest to CUDA) | {best['kernel']} ({best['size_label']}) — {best['seguru_vs_cuda']:.2f}× |")
    lines.append(f"| Worst (farthest from CUDA) | {worst['kernel']} ({worst['size_label']}) — {worst['seguru_vs_cuda']:.2f}× |")
    lines.append("")

    # Size breakdown
    lines.append("## Overhead by Input Size\n")
    lines.append("| Size | Avg SeGuRu/CUDA |")
    lines.append("|---|---|")
    for sz in ["small", "large"]:
        if sz in size_avgs:
            lines.append(f"| {sz} | {size_avgs[sz]:.2f}× |")
    lines.append("")

    # Category breakdown
    lines.append("## Results by Category\n")
    for cat in sorted(by_cat.keys()):
        rows = by_cat[cat]
        lines.append(f"### {cat.capitalize()} ({len(set(r['kernel'] for r in rows))} kernels)\n")
        lines.append(f"Average SeGuRu/CUDA ratio: **{cat_avgs[cat]:.2f}×**\n")

        header = "| Kernel | Size | SeGuRu (µs) | CUDA (µs) |"
        sep = "|---|---|---|---|"
        if pytorch_lookup:
            header += " PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |"
            sep += "---|---|"
        else:
            header += " SeGuRu/CUDA |"
            sep += "---|"

        lines.append(header)
        lines.append(sep)
        for r in rows:
            row_str = f"| {r['kernel']} | {r['size_label']} | {r['seguru_us']:.2f} | {r['cuda_us']:.2f} |"
            if pytorch_lookup:
                pt = f"{r['pytorch_us']:.2f}" if r["pytorch_us"] else "N/A"
                sp = f"{r['seguru_vs_pytorch']:.2f}×" if r["seguru_vs_pytorch"] else "N/A"
                row_str += f" {pt} | {r['seguru_vs_cuda']:.2f}× | {sp} |"
            else:
                row_str += f" {r['seguru_vs_cuda']:.2f}× |"
            lines.append(row_str)
        lines.append("")

    # Analysis
    lines.append("## Analysis\n")
    sorted_cats = sorted(cat_avgs.items(), key=lambda x: x[1])
    lines.append(f"**Most competitive categories** (lowest overhead):")
    for cat, avg in sorted_cats[:3]:
        lines.append(f"- {cat}: {avg:.2f}×")
    lines.append("")
    lines.append(f"**Highest overhead categories:**")
    for cat, avg in sorted_cats[-3:]:
        lines.append(f"- {cat}: {avg:.2f}×")
    lines.append("")

    if size_avgs.get("small") and size_avgs.get("large"):
        if size_avgs["large"] < size_avgs["small"]:
            lines.append("**Size scaling:** Overhead decreases with larger inputs, suggesting fixed launch overhead dominates at small sizes.\n")
        else:
            lines.append("**Size scaling:** Overhead is similar across input sizes.\n")

    with open(report_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_md}", file=sys.stderr)

    # ===== Terminal Table =====
    print("\n" + "=" * 80)
    print("KernelBench Level 1 Performance Comparison")
    print("=" * 80)
    print(f"\n{'Category':<20} {'Avg SeGuRu/CUDA':>16}")
    print("-" * 40)
    for cat in sorted(cat_avgs.keys()):
        print(f"{cat:<20} {cat_avgs[cat]:>15.2f}×")
    print("-" * 40)
    print(f"{'OVERALL':<20} {overall_sc:>15.2f}×")
    if overall_sp:
        print(f"{'vs PyTorch':<20} {overall_sp:>15.2f}×")
    print(f"\nBest:  {best['kernel']} ({best['size_label']}) = {best['seguru_vs_cuda']:.2f}×")
    print(f"Worst: {worst['kernel']} ({worst['size_label']}) = {worst['seguru_vs_cuda']:.2f}×")
    print("=" * 80)

if __name__ == "__main__":
    main()
