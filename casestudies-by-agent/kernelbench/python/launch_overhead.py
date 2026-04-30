#!/usr/bin/env python3.11
"""Measure pure launch overhead: empty kernel, block=1 thread=1.

Times (a) raw cudaLaunchKernel via torch.cpp_extension, and (b) SeGuRu's
Rust-side launch path through kernelbench-b. The difference isolates
SeGuRu's FFI cost.
"""
import os, pathlib, subprocess, tempfile, json, time
import numpy as np
import torch
from torch.utils.cpp_extension import load

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = REPO / "examples/target/release/kernelbench-b"
CUDA_DIR = REPO / "examples/kernelbench-b/cuda"
BUILD_DIR = REPO / "examples/kernelbench-b/cuda_build/empty"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

ITERS = 1000

def time_fn(fn, iters):
    torch.cuda.synchronize()
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us

# SeGuRu side
x = torch.rand(16, device="cuda")
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td); (tmp/"in").mkdir(); (tmp/"out").mkdir()
    x.cpu().numpy().astype(np.float32).tofile(tmp/"in/x.bin")
    env = os.environ.copy()
    env.setdefault("LD_LIBRARY_PATH", "/usr/local/cuda-13.2/lib64:/usr/lib/llvm-20/lib")
    out = subprocess.check_output(
        [str(RUNNER), "--problem", "empty", "--in-dir", str(tmp/"in"),
         "--out-dir", str(tmp/"out"), "--iters", str(ITERS), "--shape", "16"],
        env=env, text=True)
    sg_line = [l for l in out.splitlines() if l.startswith("{")][-1]
    sg_us = json.loads(sg_line)["kernel_us"]

# CUDA side (torch.cpp_extension)
mod = load(name="kbcu_empty", sources=[str(CUDA_DIR/"empty.cu")],
           build_directory=str(BUILD_DIR), extra_cuda_cflags=["-O3"], verbose=False)
cu_us = time_fn(lambda: mod.run(x), ITERS)

# Also: raw torch.cuda.Event overhead (torch op, not our kernel).
import torch.nn.functional as F
tr_us = time_fn(lambda: F.relu(x), ITERS)

print(f"Per-launch us (block=1, thread=1, empty kernel):")
print(f"  raw-CUDA  (torch.cpp_extension): {cu_us:6.2f} us")
print(f"  SeGuRu    (Rust->cuLaunchKernel): {sg_us:6.2f} us")
print(f"  torch F.relu (reference)        : {tr_us:6.2f} us")
print(f"  SeGuRu extra FFI overhead       : {sg_us - cu_us:+6.2f} us")
