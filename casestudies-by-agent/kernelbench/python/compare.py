#!/usr/bin/env python3.11
"""Phase B.full — LLM SeGuRu arm vs LLM CUDA arm, head-to-head.

SeGuRu arm: reuses the Rust binary at examples/target/release/kernelbench-b
            (outputs already validated by B.v0.1 driver.py)
CUDA arm:   compiles each .cu in cuda/ via torch.utils.cpp_extension.load
            and calls .run() directly on torch tensors (no file I/O).

Both arms are driven against the same random inputs and the same PyTorch
reference, with timing bracketed by torch.cuda.Event.
"""
from __future__ import annotations
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = REPO / "examples/target/release/kernelbench-b"
CUDA_DIR = REPO / "examples/kernelbench-b/cuda"
BUILD_DIR = REPO / "examples/kernelbench-b/cuda_build"
BUILD_DIR.mkdir(exist_ok=True)


def time_fn(fn, iters=50, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us


def dump_bin(path, t):
    t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).tofile(path)


def load_bin(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def run_seguru(problem, in_dir, out_dir, iters, shape):
    env = os.environ.copy()
    env.setdefault("LD_LIBRARY_PATH", "/usr/local/cuda-13.2/lib64:/usr/lib/llvm-20/lib")
    cmd = [str(RUNNER), "--problem", problem, "--in-dir", str(in_dir),
           "--out-dir", str(out_dir), "--iters", str(iters),
           "--shape", ",".join(str(s) for s in shape)]
    out = subprocess.check_output(cmd, env=env, text=True)
    line = [l for l in out.splitlines() if l.strip().startswith("{")][-1]
    return json.loads(line)


def compile_cuda(name):
    src = CUDA_DIR / f"{name}.cu"
    if not src.exists():
        return None, f"missing {src}"
    bdir = BUILD_DIR / name
    bdir.mkdir(parents=True, exist_ok=True)
    try:
        mod = load(
            name=f"kbcu_{name}",
            sources=[str(src)],
            build_directory=str(bdir),
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return mod, None
    except Exception as e:
        return None, f"compile error: {type(e).__name__}: {e}"


# ---------- problems ----------

def problem(name, torch_fn, ref_fn, make_x, shape, cuda_run):
    """Generic runner. Returns dicts for seguru arm and cuda arm."""
    x = make_x()
    ref = ref_fn(x)
    torch_us = time_fn(lambda: torch_fn(x))

    # SeGuRu arm (file-based, via rust runner)
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td); (tmp / "in").mkdir(); (tmp / "out").mkdir()
        dump_bin(tmp / "in/x.bin", x)
        sg = run_seguru(name, tmp / "in", tmp / "out", iters=50, shape=shape)
        y_sg = torch.from_numpy(load_bin(tmp / "out/y.bin", tuple(x.shape))).cuda()
        sg_err = float((y_sg - ref).abs().max())
        sg_row = dict(arm="seguru", problem=name, torch_us=torch_us,
                      impl_us=sg["kernel_us"], max_err=sg_err)

    # CUDA arm (in-process torch extension)
    mod, err = compile_cuda(name)
    if mod is None:
        cu_row = dict(arm="cuda", problem=name, torch_us=torch_us,
                      impl_us=float("inf"), max_err=float("inf"), error=err)
    else:
        try:
            y_cu = cuda_run(mod, x)
            cu_err = float((y_cu - ref).abs().max())
            cu_us = time_fn(lambda: cuda_run(mod, x))
            cu_row = dict(arm="cuda", problem=name, torch_us=torch_us,
                          impl_us=cu_us, max_err=cu_err)
        except Exception as e:
            cu_row = dict(arm="cuda", problem=name, torch_us=torch_us,
                          impl_us=float("inf"), max_err=float("inf"),
                          error=f"{type(e).__name__}: {e}")
    return sg_row, cu_row


def leaky_relu_case():
    B, D = 4096, 393216
    return problem(
        name="leaky_relu",
        torch_fn=lambda x: F.leaky_relu(x, negative_slope=0.01),
        ref_fn=lambda x: F.leaky_relu(x, negative_slope=0.01),
        make_x=lambda: (torch.rand(B, D, device="cuda") * 2 - 1),
        shape=[B, D],
        cuda_run=lambda mod, x: mod.run(x, 0.01),
    )


def tanh_case():
    B, D = 4096, 393216
    return problem(
        name="tanh",
        torch_fn=lambda x: torch.tanh(x),
        ref_fn=lambda x: torch.tanh(x),
        make_x=lambda: (torch.rand(B, D, device="cuda") * 2 - 1),
        shape=[B, D],
        cuda_run=lambda mod, x: mod.run(x),
    )


def rms_norm_case():
    B, C, H, W = 112, 64, 512, 512
    def torch_fn(x):
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-5)
        return x / rms
    return problem(
        name="rms_norm",
        torch_fn=torch_fn,
        ref_fn=torch_fn,
        make_x=lambda: torch.rand(B, C, H, W, device="cuda"),
        shape=[B, C, H, W],
        cuda_run=lambda mod, x: mod.run(x, 1e-5),
    )


def fmt_row(r):
    corr = "OK" if r["max_err"] < 1e-3 else "FAIL"
    impl_us = f"{r['impl_us']:9.2f}us" if r["impl_us"] != float("inf") else "   FAILED"
    speedup = r["torch_us"] / r["impl_us"] if r["impl_us"] not in (0, float("inf")) else 0
    extra = f"  [{r.get('error','')[:50]}]" if "error" in r else ""
    return (f"  {r['arm']:6s} {r['problem']:12s} "
            f"torch={r['torch_us']:8.2f}us  impl={impl_us}  "
            f"speedup={speedup:5.2f}x  err={r['max_err']:.2e}  {corr}{extra}")


def main():
    assert RUNNER.exists(), "build first: cargo build --release -p kernelbench-b"
    all_rows = []
    for case in [leaky_relu_case, tanh_case, rms_norm_case]:
        sg, cu = case()
        print(fmt_row(sg))
        print(fmt_row(cu))
        all_rows.extend([sg, cu])

    print("\n=== summary per arm ===")
    for arm in ["seguru", "cuda"]:
        rows = [r for r in all_rows if r["arm"] == arm]
        n = len(rows)
        correct = [r for r in rows if r["max_err"] < 1e-3]
        fast1 = [r for r in correct if r["torch_us"] / r["impl_us"] >= 1.0]
        fast2 = [r for r in correct if r["torch_us"] / r["impl_us"] >= 2.0]
        print(f"  {arm:6s}  fast_0={len(correct)*100//n:3d}%  "
              f"fast_1={len(fast1)*100//n:3d}%  fast_2={len(fast2)*100//n:3d}%  "
              f"({len(correct)}/{n} correct)")


if __name__ == "__main__":
    main()
