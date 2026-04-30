#!/usr/bin/env python3.11
"""Phase B v0.1 driver — LLM-generated SeGuRu kernels."""
from __future__ import annotations
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = REPO / "examples/target/release/kernelbench-b"


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
    return s.elapsed_time(e) * 1000.0 / iters


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


def problem_leaky_relu(tmp, iters):
    B, D = 4096, 393216
    x = torch.rand(B, D, device="cuda") * 2 - 1
    ref = F.leaky_relu(x, negative_slope=0.01)
    t_us = time_torch(lambda: F.leaky_relu(x, negative_slope=0.01), iters)
    dump_bin(tmp / "in/x.bin", x)
    res = run_seguru("leaky_relu", tmp / "in", tmp / "out", iters, [B, D])
    y = torch.from_numpy(load_bin(tmp / "out/y.bin", (B, D))).cuda()
    err = float((y - ref).abs().max())
    return dict(problem="leaky_relu", shape=[B, D], torch_us=t_us,
                seguru_us=res["kernel_us"], max_err=err)


def problem_tanh(tmp, iters):
    B, D = 4096, 393216
    x = torch.rand(B, D, device="cuda") * 2 - 1
    ref = torch.tanh(x)
    t_us = time_torch(lambda: torch.tanh(x), iters)
    dump_bin(tmp / "in/x.bin", x)
    res = run_seguru("tanh", tmp / "in", tmp / "out", iters, [B, D])
    y = torch.from_numpy(load_bin(tmp / "out/y.bin", (B, D))).cuda()
    err = float((y - ref).abs().max())
    return dict(problem="tanh", shape=[B, D], torch_us=t_us,
                seguru_us=res["kernel_us"], max_err=err)


def problem_rms_norm(tmp, iters):
    B, C, H, W = 112, 64, 512, 512
    eps = 1e-5
    x = torch.rand(B, C, H, W, device="cuda")
    def fn():
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)
        return x / rms
    ref = fn()
    t_us = time_torch(fn, iters)
    dump_bin(tmp / "in/x.bin", x)
    res = run_seguru("rms_norm", tmp / "in", tmp / "out", iters, [B, C, H, W])
    y = torch.from_numpy(load_bin(tmp / "out/y.bin", (B, C, H, W))).cuda()
    err = float((y - ref).abs().max())
    return dict(problem="rms_norm", shape=[B, C, H, W], torch_us=t_us,
                seguru_us=res["kernel_us"], max_err=err)


PROBLEMS = [problem_leaky_relu, problem_tanh, problem_rms_norm]


def main():
    assert RUNNER.exists(), "build first: cargo build --release -p kernelbench-b"
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "in").mkdir()
        (tmp / "out").mkdir()
        for fn in PROBLEMS:
            try:
                r = fn(tmp, iters=50)
                r["correct"] = r["max_err"] < 1e-3
                r["speedup"] = r["torch_us"] / r["seguru_us"] if r["seguru_us"] > 0 else 0.0
                rows.append(r)
                print(f"{r['problem']:12s} shape={r['shape']!s:22s} "
                      f"torch={r['torch_us']:8.2f}us  seguru={r['seguru_us']:9.2f}us  "
                      f"speedup={r['speedup']:5.2f}x  err={r['max_err']:.2e}  "
                      f"{'OK' if r['correct'] else 'FAIL'}")
            except Exception as e:
                print(f"{fn.__name__:28s} ERROR: {e}")
                rows.append(dict(problem=fn.__name__, correct=False, speedup=0, max_err=float('inf')))

    n = len(rows)
    f0 = sum(1 for r in rows if r.get("correct")) / n
    f1 = sum(1 for r in rows if r.get("correct") and r.get("speedup", 0) >= 1.0) / n
    f2 = sum(1 for r in rows if r.get("correct") and r.get("speedup", 0) >= 2.0) / n
    print()
    print(f"fast_0 (correct):       {f0*100:5.1f}%  ({sum(1 for r in rows if r.get('correct'))}/{n})")
    print(f"fast_1 (correct & 1x+): {f1*100:5.1f}%")
    print(f"fast_2 (correct & 2x+): {f2*100:5.1f}%")


if __name__ == "__main__":
    main()
