#!/usr/bin/env python3.11
"""Plots benchdata/all.csv: the cost of SeGuRu against CUDA, with and without
bounds checks.

Two figures:

  bench_ratio.png   a box plot of the SeGuRu/CUDA ratio per suite (the shape of
                    the result) beside a per-workload bar chart (the detail).
  bench_times.png   absolute times per suite, log scale, SeGuRu against every
                    baseline that suite has.

Only workloads that actually have a CUDA baseline can appear in the ratio plots;
kernelbench has none, so it is excluded there and shown only in the time plot.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "benchdata" / "all.csv"

# Bars are ratios against CUDA, so lower is better and 1.0 is parity.
STOCK_C = "#4C72B0"
NOBC_C = "#DD8452"
LABEL = {"stock": "SeGuRu, bounds checks on",
         "nobc": "SeGuRu, DISABLE_GPU_BOUND_CHECK"}

def load():
    df = pd.read_csv(CSV)
    # Only durations are comparable as a ratio; throughput rows are kept for the
    # time plot's annotations but never mixed into a ratio.
    return df[df.metric == "time"].copy()


def ratios(df):
    """SeGuRu-vs-CUDA ratio per (variant, suite, workload, parameter).

    The baseline is always the `cuda` implementation measured in the *stock*
    run: it is the same CUDA binary in both variants, and DISABLE_GPU_BOUND_CHECK
    changes only SeGuRu codegen. Using the stock CUDA number for both keeps the
    two bars comparable against one fixed reference.
    """
    key = ["suite", "workload", "parameter"]
    cuda = (df[(df.implementation == "cuda") & (df.variant == "stock")]
            .set_index(key)["value"])
    out = []
    for variant in ("stock", "nobc"):
        sg = df[(df.implementation == "seguru") & (df.variant == variant)]
        for _, r in sg.iterrows():
            k = (r.suite, r.workload, r.parameter)
            if k in cuda.index:
                base = cuda.loc[k]
                if np.isscalar(base) and base > 0:
                    out.append(dict(variant=variant, suite=r.suite,
                                    workload=r.workload, parameter=r.parameter,
                                    seguru=r.value, cuda=base,
                                    ratio=r.value / base, units=r.units))
    return pd.DataFrame(out)


def fig_box(rat, path):
    """Shape of the result: how the SeGuRu/CUDA ratio is distributed per suite."""
    have_nobc = (rat.variant == "nobc").any()
    variants = ["stock", "nobc"] if have_nobc else ["stock"]
    suites = sorted(rat.suite.unique())
    fig, ax = plt.subplots(figsize=(1.7 * len(suites) + 4, 5.5))
    width = 0.8 / len(variants)
    for vi, v in enumerate(variants):
        data = [rat[(rat.suite == s) & (rat.variant == v)].ratio.values for s in suites]
        pos = np.arange(len(suites)) + (vi - (len(variants) - 1) / 2) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                        manage_ticks=False, showfliers=True,
                        medianprops=dict(color="black", lw=1.5),
                        flierprops=dict(marker="o", ms=3.5, alpha=.65))
        for b in bp["boxes"]:
            b.set_facecolor(STOCK_C if v == "stock" else NOBC_C)
            b.set_alpha(.8)
        # A legend proxy, because boxplot() does not label itself.
        ax.bar(np.nan, np.nan, color=STOCK_C if v == "stock" else NOBC_C, alpha=.8,
               label=LABEL[v])
    ax.axhline(1.0, color="k", lw=1.2, ls="--", alpha=.8)
    ax.set_xticks(np.arange(len(suites)))
    ax.set_xticklabels([f"{s}\n({(rat.suite == s).sum() // len(variants)} workloads)"
                        for s in suites])
    ax.set_ylabel("SeGuRu / CUDA   (1.0 = parity, lower is better)")
    ax.set_title("Cost of safe Rust on SeGuRu, per benchmark suite (A100)\n"
                 "box = quartiles, whiskers = 1.5 IQR, dots = individual workloads",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


def fig_detail(rat, path):
    """Every workload at every problem size."""
    have_nobc = (rat.variant == "nobc").any()
    det = rat[rat.variant == "stock"].sort_values(["suite", "workload", "parameter"])
    n = len(det)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.19 * n + 1.6)))
    labels = [f"{r.suite} / {r.workload}  [{r.parameter}]" for _, r in det.iterrows()]
    y = np.arange(n)
    h = 0.40 if have_nobc else 0.72
    ax.barh(y + (h / 2 if have_nobc else 0), det.ratio.values, height=h,
            color=STOCK_C, label=LABEL["stock"], zorder=3)
    if have_nobc:
        nb = rat[rat.variant == "nobc"].set_index(["suite", "workload", "parameter"]).ratio
        vals = [nb.get((r.suite, r.workload, r.parameter), np.nan) for _, r in det.iterrows()]
        ax.barh(y - h / 2, vals, height=h, color=NOBC_C, label=LABEL["nobc"], zorder=3)
    ax.axvline(1.0, color="k", lw=1.2, ls="--", alpha=.8, zorder=4)
    # Separator between suites, so the groups read at a glance.
    suite_of = list(det.suite)
    for i in range(1, n):
        if suite_of[i] != suite_of[i - 1]:
            ax.axhline(i - 0.5, color="0.4", lw=.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("SeGuRu / CUDA   (1.0 = parity, lower is better)")
    ax.set_title("Per workload and problem size", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=.3, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


def fig_times(df, path):
    suites = sorted(df.suite.unique())
    fig, axes = plt.subplots(len(suites), 1, figsize=(14, 3.4 * len(suites)))
    if len(suites) == 1:
        axes = [axes]
    for ax, s in zip(axes, suites):
        d = df[(df.suite == s) & (df.variant == "stock")]
        items = sorted({(w, p) for w, p in zip(d.workload, d.parameter)})
        impls = [i for i in ["seguru", "cuda", "cuda_upstream_tuning", "cub",
                             "thrust", "cublas", "cpu"] if i in set(d.implementation)]
        x = np.arange(len(items))
        w = 0.8 / max(1, len(impls))
        for i, im in enumerate(impls):
            sub = d[d.implementation == im].set_index(["workload", "parameter"])["value"]
            vals = [sub.get(k, np.nan) for k in items]
            vals = [v if np.isscalar(v) else np.nan for v in vals]
            ax.bar(x + (i - (len(impls) - 1) / 2) * w, vals, width=w, label=im)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}\n{b}" for a, b in items], fontsize=6, rotation=45,
                           ha="right")
        unit = d.units.mode().iat[0] if len(d) else ""
        ax.set_ylabel(f"time ({unit}), log")
        ax.set_title(s)
        ax.legend(fontsize=7, ncol=len(impls))
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


def write_results_md(df, rat, path):
    """The human-readable companion to all.csv: every workload, both variants."""
    key = ["suite", "workload", "parameter"]
    piv = rat.pivot_table(index=key, columns="variant", values=["seguru", "ratio"])
    cuda = (df[(df.implementation == "cuda") & (df.variant == "stock")]
            .set_index(key)["value"])
    units = df.set_index(key)["units"]

    lines = ["# Case-study performance data",
             "",
             "Generated by `plot-bench.py` from `benchdata/all.csv`, which is produced by",
             "`./bench-all.sh`. Every row is a mean over the harness's own iteration count",
             "on an idle A100 80GB PCIe (driver 580.159.03, CUDA 13.3).",
             "",
             "`SeGuRu` is the agent-written safe-Rust kernel. `no-bc` is the identical",
             "source rebuilt with `DISABLE_GPU_BOUND_CHECK=true`, which makes",
             "`rustc_codegen_gpu` skip the per-access slice bounds check (verified: 445",
             "`setp.gt.u64` in the polybench PTX drop to 0 across the same 57 kernels).",
             "`CUDA` is the hand-written CUDA C++ baseline computing the same thing.",
             "",
             "A ratio below 1.0 means the safe-Rust kernel is faster.",
             ""]

    for suite in sorted(rat.suite.unique()):
        sub = piv[piv.index.get_level_values("suite") == suite]
        lines += [f"## {suite}", "",
                  "| workload | parameter | SeGuRu | no-bc | CUDA | SG/CUDA | no-bc/CUDA | units |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | :-- |"]
        for k in sub.index:
            r = sub.loc[k]
            u = units.loc[k]
            u = u.iat[0] if hasattr(u, "iat") else u
            def g(col, v):
                try:
                    x = r[(col, v)]
                    return f"{x:.3f}" if x == x else "--"
                except KeyError:
                    return "--"
            lines.append(f"| {k[1]} | {k[2]} | {g('seguru','stock')} | {g('seguru','nobc')} "
                         f"| {cuda.loc[k]:.3f} | {g('ratio','stock')} | {g('ratio','nobc')} | {u} |")
        lines.append("")

    # Workloads with no CUDA baseline at all still deserve a record.
    noref = df[(df.variant == "stock") & (df.metric == "time")
               & (~df.set_index(key).index.isin(cuda.index))]
    if len(noref):
        lines += ["## No CUDA baseline (SeGuRu only)", "",
                  "| suite | workload | parameter | SeGuRu | no-bc | units |",
                  "| --- | --- | --- | ---: | ---: | :-- |"]
        nb = (df[(df.variant == "nobc") & (df.implementation == "seguru")
                 & (df.metric == "time")].set_index(key)["value"])
        seen = set()
        for _, r in noref[noref.implementation == "seguru"].iterrows():
            k = (r.suite, r.workload, r.parameter)
            if k in seen:
                continue
            seen.add(k)
            v = nb.get(k, float("nan"))
            v = v if np.isscalar(v) else float("nan")
            lines.append(f"| {r.suite} | {r.workload} | {r.parameter} | {r.value:.3f} "
                         f"| {'--' if v != v else f'{v:.3f}'} | {r.units} |")
        lines.append("")

    lines += ["## Summary: median ratio per suite", "",
              "| suite | workloads | median SG/CUDA | median no-bc/CUDA | bounds-check cost |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for suite in sorted(rat.suite.unique()):
        a = rat[(rat.suite == suite) & (rat.variant == "stock")].ratio.median()
        b = rat[(rat.suite == suite) & (rat.variant == "nobc")].ratio.median()
        n = (rat.suite == suite).sum() // 2
        lines.append(f"| {suite} | {n} | {a:.3f} | {b:.3f} | {(a / b - 1) * 100:+.1f}% |")
    lines.append("")

    path.write_text("\n".join(lines))
    print("wrote", path)


def main():
    if not CSV.exists():
        sys.exit(f"{CSV} not found - run ./bench-all.sh first")
    df = load()
    rat = ratios(df)
    rat.sort_values(["suite", "workload", "parameter", "variant"]).to_csv(
        HERE / "benchdata" / "ratios.csv", index=False, float_format="%.6f")
    print("ratio rows:", len(rat))
    write_results_md(df, rat, HERE / "benchdata" / "RESULTS.md")
    fig_box(rat, HERE / "benchdata" / "bench_box.png")
    fig_detail(rat, HERE / "benchdata" / "bench_detail.png")
    fig_times(df, HERE / "benchdata" / "bench_times.png")
    write_latex(df, rat)




# ---------------------------------------------------------------------------
# LaTeX tables for the paper appendix (PolyBench and KernelBench only).
# Written into the seguru-paper working copy if it is present next door, so the
# paper never carries hand-transcribed numbers.
# ---------------------------------------------------------------------------

PAPER = HERE.parent / "seguru-paper" / "src"

HDR = ("% GENERATED by casestudies-by-agent-v2/plot-bench.py from benchdata/all.csv.\n"
       "% Do not edit by hand: re-run ./bench-all.sh && python3.11 plot-bench.py.\n")


def _fmt_size(p):
    """PolyBench size tokens like 4096^3 or 8192^2xt10 into math mode."""
    p = p.replace("xt", r", $t{=}$")
    out, i = [], 0
    while i < len(p):
        if p[i] == "^":
            out.append("$^{" + p[i + 1] + "}$")
            i += 2
        else:
            out.append(p[i])
            i += 1
    return "".join(out)


def write_latex(df, rat):
    if not PAPER.is_dir():
        print("no seguru-paper checkout next door; skipping LaTeX")
        return

    # ---- PolyBench -------------------------------------------------------
    key = ["suite", "workload", "parameter"]
    cuda = (df[(df.implementation == "cuda") & (df.variant == "stock")]
            .set_index(key)["value"])
    st = rat[(rat.suite == "polybench") & (rat.variant == "stock")].copy()
    nb = rat[(rat.suite == "polybench") & (rat.variant == "nobc")].set_index(key).ratio
    st["size_key"] = st.seguru
    st = st.sort_values(["workload", "size_key"])

    body = []
    for _, r in st.iterrows():
        k = (r.suite, r.workload, r.parameter)
        n = nb.get(k, np.nan)
        n = n if np.isscalar(n) else np.nan
        body.append("{} & {} & {:.1f} & {:.1f} & {:.2f} & {}".format(
            r.workload.replace("_", r"\_"), _fmt_size(r.parameter),
            r.seguru, cuda.loc[k], r.ratio,
            "--" if n != n else f"{n:.2f}"))

    half = (len(body) + 1) // 2
    hdr = (r"Kernel & Size & \sys{} & CUDA & $\frac{\text{\sys{}}}{\text{CUDA}}$ & "
           r"$\frac{\text{no-bc}}{\text{CUDA}}$")
    tex = [HDR, r"\begin{table*}[t]", r"\centering",
           r"\caption{PolyBench: every agent-written \sys{} kernel against the "
           r"hand-written CUDA C++ mirror computing the same thing (A100). "
           r"\emph{no-bc} is the identical Rust source rebuilt with bounds checks "
           r"disabled; a ratio below $1.0$ means \sys{} is faster."
           r"\label{tab:appendix:polybench}}",
           r"\scriptsize",
           r"\begin{tabular}{@{}llrrrr@{\hskip 2.2em}llrrrr@{}}",
           r"\toprule",
           hdr + " & " + hdr + r" \\",
           " & ".join([r" & & ($\mu$s) & ($\mu$s) & &"] * 2) + r" \\",
           r"\midrule"]
    for i in range(half):
        right = body[i + half] if i + half < len(body) else "& & & & &"
        tex.append(f"{body[i]} & {right} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (PAPER / "tab_appendix_polybench.tex").write_text("\n".join(tex))
    print("wrote", PAPER / "tab_appendix_polybench.tex")

    # ---- KernelBench -----------------------------------------------------
    # `df` has been filtered to durations; KernelBench also reports achieved
    # bandwidth, so re-read the unfiltered file for it.
    raw = pd.read_csv(CSV)
    kb = raw[(raw.suite == "kernelbench") & (raw.variant == "stock")].copy()
    extra = kb.parameter.str.extract(r"/(.+)$")[0]
    kb["workload"] = np.where(extra.notna(),
                              kb.workload + " (" + extra.fillna("") + ")",
                              kb.workload)
    kb["parameter"] = kb.parameter.str.replace(r"/.*$", "", regex=True)
    shapes = sorted(kb.parameter.unique(), key=lambda s: int(s.split("x")[0]))
    t = kb[kb.metric == "time"].set_index(["workload", "parameter"])["value"]
    g = kb[kb.metric == "throughput"].set_index(["workload", "parameter"])["value"]
    ops = list(dict.fromkeys(kb.workload))

    tex = [HDR, r"\begin{table}[t]", r"\centering",
           r"\caption{KernelBench operators in \sys{}. These are PyTorch "
           r"operators with no hand-written CUDA mirror, so we report achieved "
           r"bandwidth instead of a ratio; the A100's measured copy roof is "
           r"about $1550$\,GB/s.\label{tab:appendix:kernelbench}}",
           r"\scriptsize",
           r"\begin{tabular}{@{}l" + "rr" * len(shapes) + r"@{}}",
           r"\toprule",
           "Operator & " + " & ".join(
               r"\multicolumn{2}{c}{" + s.replace("x", r"$\times$") + "}"
               for s in shapes) + r" \\",
           "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(shapes))),
           " & " + " & ".join(r"($\mu$s) & GB/s" for _ in shapes) + r" \\",
           r"\midrule"]
    for op in ops:
        cells = []
        for s in shapes:
            tv, gv = t.get((op, s), np.nan), g.get((op, s), np.nan)
            tv = tv if np.isscalar(tv) else np.nan
            gv = gv if np.isscalar(gv) else np.nan
            cells += ["--" if tv != tv else f"{tv:.1f}",
                      "--" if gv != gv else f"{gv:.0f}"]
        tex.append(op.replace("_", r"\_").replace("=", r"{=}")
                   + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER / "tab_appendix_kernelbench.tex").write_text("\n".join(tex))
    print("wrote", PAPER / "tab_appendix_kernelbench.tex")


if __name__ == "__main__":
    main()
