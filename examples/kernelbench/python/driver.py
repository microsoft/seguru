#!/usr/bin/env python3.11
"""KernelBench-style driver for the SeGuRu runner.

For each problem:
  1. Generate random inputs with torch.
  2. Time the PyTorch reference implementation (CUDA events).
  3. Dump inputs + expected output to `.bin` files.
  4. Spawn `kernelbench run --problem X ...` (built SeGuRu binary).
  5. Read the SeGuRu output; compute max-abs-err vs torch reference.
  6. Print per-problem row + aggregate fast_0/fast_1/fast_2.

Env: LD_LIBRARY_PATH must include /usr/local/cuda-13.2/lib64 for pytorch+CUDA.
Run: python3.11 driver.py
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[3]

def target_dir():
    env_target = os.environ.get("CARGO_TARGET_DIR")
    if env_target:
        return pathlib.Path(env_target)
    local_target = REPO / "examples/target"
    if local_target.exists():
        return local_target
    parts = REPO.parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        shared_target = pathlib.Path(*parts[:idx]) / "examples/target"
        if shared_target.exists():
            return shared_target
    return local_target


RUNNER = target_dir() / "release/kernelbench"
SCRATCH_ROOT = target_dir() / "kernelbench-driver-tmp"

ATOL = 1e-4


def time_torch(fn, iters=100, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us


def dump_bin(path, t: torch.Tensor):
    arr = t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    arr.tofile(path)


def load_bin(path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    return arr.reshape(shape)


def run_seguru(problem, in_dir, out_dir, iters, shape):
    cmd = [
        str(RUNNER), "run",
        "--problem", problem,
        "--in-dir", str(in_dir),
        "--out-dir", str(out_dir),
        "--iters", str(iters),
        "--shape", ",".join(str(s) for s in shape),
    ]
    env = os.environ.copy()
    env.setdefault("LD_LIBRARY_PATH",
                   "/usr/local/cuda-13.2/lib64:/usr/lib/llvm-20/lib")
    # last non-empty line is the JSON
    out = subprocess.check_output(cmd, env=env, text=True)
    line = [l for l in out.splitlines() if l.strip().startswith("{")][-1]
    return json.loads(line)


def problem_relu(tmp, iters):
    M, N = 4096, 16384
    x = torch.randn(M, N, device="cuda")
    ref = F.relu(x)
    torch_us = time_torch(lambda: F.relu(x), iters=iters)
    dump_bin(tmp / "in/x.bin", x)
    res = run_seguru("relu", tmp / "in", tmp / "out", iters, [M * N])
    y = load_bin(tmp / "out/y.bin", (M, N))
    err = float((torch.from_numpy(y).to("cuda") - ref).abs().max())
    return dict(problem="relu", shape=[M, N], torch_us=torch_us,
                seguru_us=res["kernel_us"], max_err=err)


def problem_sigmoid(tmp, iters):
    M, N = 4096, 16384
    x = torch.randn(M, N, device="cuda")
    ref = torch.sigmoid(x)
    torch_us = time_torch(lambda: torch.sigmoid(x), iters=iters)
    dump_bin(tmp / "in/x.bin", x)
    res = run_seguru("sigmoid", tmp / "in", tmp / "out", iters, [M * N])
    y = load_bin(tmp / "out/y.bin", (M, N))
    err = float((torch.from_numpy(y).to("cuda") - ref).abs().max())
    return dict(problem="sigmoid", shape=[M, N], torch_us=torch_us,
                seguru_us=res["kernel_us"], max_err=err)


def problem_softmax(tmp, iters):
    M, N = 4096, 4096  # N must match compile-time SOFTMAX_N
    x = torch.randn(M, N, device="cuda")
    ref = F.softmax(x, dim=1)
    torch_us = time_torch(lambda: F.softmax(x, dim=1), iters=iters)
    dump_bin(tmp / "in/x.bin", x)
    res = run_seguru("softmax", tmp / "in", tmp / "out", iters, [M, N])
    y = load_bin(tmp / "out/y.bin", (M, N))
    err = float((torch.from_numpy(y).to("cuda") - ref).abs().max())
    return dict(problem="softmax", shape=[M, N], torch_us=torch_us,
                seguru_us=res["kernel_us"], max_err=err)


def problem_matmul(tmp, iters):
    N = 4096
    a = torch.randn(N, N, device="cuda")
    b = torch.randn(N, N, device="cuda")
    ref = a @ b
    torch_us = time_torch(lambda: a @ b, iters=iters)
    dump_bin(tmp / "in/a.bin", a)
    dump_bin(tmp / "in/b.bin", b)
    res = run_seguru("matmul", tmp / "in", tmp / "out", iters, [N])
    c = load_bin(tmp / "out/c.bin", (N, N))
    err = float((torch.from_numpy(c).to("cuda") - ref).abs().max())
    return dict(problem="matmul", shape=[N, N], torch_us=torch_us,
                seguru_us=res["kernel_us"], max_err=err)


def problem_avg_pool1d(tmp, iters):
    B, C, L = 64, 128, 65536
    K, S, P = 8, 1, 4
    x = torch.rand(B, C, L, device="cuda")
    ref = F.avg_pool1d(x, kernel_size=K, stride=S, padding=P)
    torch_us = time_torch(lambda: F.avg_pool1d(x, kernel_size=K, stride=S, padding=P), iters=iters)
    dump_bin(tmp / "in/x.bin", x)
    out_len = ref.shape[-1]
    res = run_seguru("avg_pool1d", tmp / "in", tmp / "out", iters, [B, C, L, K, S, P])
    y = load_bin(tmp / "out/y.bin", (B, C, out_len))
    err = float((torch.from_numpy(y).to("cuda") - ref).abs().max())
    return dict(problem="avg_pool1d", shape=[B, C, L], torch_us=torch_us,
                seguru_us=res["kernel_us"], max_err=err)


PROBLEMS = [
    problem_relu,
    problem_sigmoid,
    problem_softmax,
    problem_matmul,
    problem_avg_pool1d,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--only", help="comma-separated subset")
    ap.add_argument("--problem", help="single problem or comma-separated subset")
    args = ap.parse_args()

    assert RUNNER.exists(), (
        f"runner not found at {RUNNER}; build with `cargo build --release -p kernelbench`")

    selected = args.problem or args.only
    wanted = set(selected.split(",")) if selected else None
    rows = []
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as td:
        tmp = pathlib.Path(td)
        (tmp / "in").mkdir()
        (tmp / "out").mkdir()
        for fn in PROBLEMS:
            name = fn.__name__.replace("problem_", "")
            if wanted and name not in wanted:
                continue
            r = fn(tmp, args.iters)
            r["correct"] = r["max_err"] < ATOL * max(1.0, r["torch_us"])  # scale-free ok for bench
            r["correct"] = r["max_err"] < 1e-3  # simple threshold
            r["speedup"] = r["torch_us"] / r["seguru_us"] if r["seguru_us"] > 0 else 0.0
            rows.append(r)
            print(f"{r['problem']:10s} shape={r['shape']!s:20s} "
                  f"torch={r['torch_us']:8.2f}us  seguru={r['seguru_us']:8.2f}us  "
                  f"speedup={r['speedup']:5.2f}x  err={r['max_err']:.1e}  "
                  f"{'OK' if r['correct'] else 'FAIL'}")

    n = len(rows)
    if n == 0:
        return
    fast_0 = sum(1 for r in rows if r["correct"]) / n
    fast_1 = sum(1 for r in rows if r["correct"] and r["speedup"] >= 1.0) / n
    fast_2 = sum(1 for r in rows if r["correct"] and r["speedup"] >= 2.0) / n
    print()
    print(f"fast_0 (correct):        {fast_0*100:5.1f}%  ({sum(r['correct'] for r in rows)}/{n})")
    print(f"fast_1 (correct & 1x+):  {fast_1*100:5.1f}%")
    print(f"fast_2 (correct & 2x+):  {fast_2*100:5.1f}%")


if __name__ == "__main__":
    main()
