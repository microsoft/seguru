#!/usr/bin/env python3.11
"""Phase C — 4-arm comparison for KernelBench Level 2 (fused operators).

Same structure as kernelbench-b/python/compare2.py, but problems have multiple
inputs (x, W, b for GEMM; x, W, conv_bias, extra_bias for Conv2d). The SeGuRu
arm receives all inputs via separate .bin files dumped into in_dir.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, tempfile, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = REPO / "examples/target/release/kernelbench-c"
CUDA_DIR = REPO / "examples/kernelbench-c/cuda"
BUILD_DIR = REPO / "examples/kernelbench-c/cuda_build"
BUILD_DIR.mkdir(exist_ok=True)


def time_fn(fn, iters=50, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
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
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
            verbose=False,
        )
        return mod, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


# ---------- problem registry ----------
# Each entry describes:
#   make_inputs(): returns dict of CUDA tensors to dump; keys become {name}.bin
#   torch_fn(ins): runs the reference; takes the same dict
#   cuda_run(mod, ins): runs the raw CUDA extension
#   in_shape: shape tuple passed to the SeGuRu runner (problem-specific schema)
#   out_shape: output tensor shape
#   atol: absolute tolerance for correctness

def _gemm_mul_lrelu():
    M, K, N = 1024, 8192, 8192
    multiplier = 2.0
    negative_slope = 0.1
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        # Match nn.Linear initialization (uniform in roughly [-1/sqrt(K), 1/sqrt(K)])
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = y * multiplier
        return F.leaky_relu(y, negative_slope)
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], multiplier, negative_slope)
    return dict(
        make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
        in_shape=[M, K, N], out_shape=[M, N], atol=5e-3,
    )


def _conv_relu_hardswish():
    # 128 × 8 × 128 × 128, 3×3 conv, 64 out-ch, no padding → 126 × 126 output.
    B, Cin, H, W = 128, 8, 128, 128
    Cout, Kh, Kw = 64, 3, 3
    Ho, Wo = H - Kh + 1, W - Kw + 1
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(B, Cin, H, W, device="cuda")
        lim = 1.0 / ((Cin * Kh * Kw) ** 0.5)
        weight = (torch.rand(Cout, Cin, Kh, Kw, device="cuda") * 2 - 1) * lim
        conv_bias = (torch.rand(Cout, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=weight, b=conv_bias)
    def torch_fn(ins):
        y = F.conv2d(ins["x"], ins["W"], ins["b"])
        y = F.relu(y)
        y = y * torch.clamp((y + 3) / 6, 0, 1)
        return y
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"])
    return dict(
        make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
        in_shape=[B, Cin, H, W, Cout, Kh, Kw], out_shape=[B, Cout, Ho, Wo], atol=5e-3,
    )


def _matmul_mish_mish():
    M, K, N = 1024, 8192, 8192
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = F.mish(y); y = F.mish(y)
        return y
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"])
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _matmul_scale_resadd():
    M, K, N = 16384, 4096, 4096
    scaling = 0.5
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        orig = y.clone().detach()
        return y * scaling + orig
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], scaling)
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _gemm_scale_htanh_gelu():
    M, K, N = 2048, 8192, 8192
    scaling, hmin, hmax = 0.5, -2.0, 2.0
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = y * scaling
        y = torch.clamp(y, hmin, hmax)
        return F.gelu(y)
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], scaling, hmin, hmax)
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _matmul_sigmoid_sum():
    # Large inner + row-wise sum => output is [M, 1]
    M, K, N = 128, 32768, 32768
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = torch.sigmoid(y)
        return torch.sum(y, dim=1, keepdim=True)
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"])
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, 1], atol=2e-1)

def _matmul_swish_scaling():
    M, K, N = 128, 32768, 32768
    scaling_factor = 2.0
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        z = ins["x"] @ ins["W"].T + ins["b"]
        return z * torch.sigmoid(z) * scaling_factor
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"])
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _gemm_relu_div():
    M, K, N = 1024, 8192, 8192
    divisor = 2.0
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = F.relu(y)
        return y / divisor
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], divisor)
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _conv_relu_biasadd():
    # 128 × 64 × 128 × 128, 3×3 conv, 128 out-ch, bias [128,1,1]
    B, Cin, H, W = 128, 64, 128, 128
    Cout, Kh, Kw = 128, 3, 3
    Ho, Wo = H - Kh + 1, W - Kw + 1
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(B, Cin, H, W, device="cuda")
        lim = 1.0 / ((Cin * Kh * Kw) ** 0.5)
        weight = (torch.rand(Cout, Cin, Kh, Kw, device="cuda") * 2 - 1) * lim
        conv_bias = (torch.rand(Cout, device="cuda") * 2 - 1) * lim
        extra_bias = torch.randn(Cout, 1, 1, device="cuda") * lim
        return dict(x=x, W=weight, b=conv_bias, bias2=extra_bias)
    def torch_fn(ins):
        y = F.conv2d(ins["x"], ins["W"], ins["b"])
        y = F.relu(y)
        return y + ins["bias2"]
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], ins["bias2"])
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[B, Cin, H, W, Cout, Kh, Kw], out_shape=[B, Cout, Ho, Wo], atol=5e-3)


def _matmul_sub_mul_relu():
    M, K, N = 1024, 8192, 8192
    sub_val, mul_val = 2.0, 1.5
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = y - sub_val
        y = y * mul_val
        return F.relu(y)
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], sub_val, mul_val)
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _gemm_add_relu():
    M, K, N = 1024, 8192, 8192
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        # Reference: Linear bias=False, then add extra bias of shape [N] sampled from randn.
        b = torch.randn(N, device="cuda")
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        return F.relu(y)
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"])
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _matmul_div_gelu():
    M, K, N = 1024, 8192, 8192
    divisor = 10.0
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = y / divisor
        return F.gelu(y, approximate="tanh")
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], divisor)
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)

def _matmul_min_subtract():
    M, K, N = 128, 16384, 16384
    constant = 2.0
    def make_inputs():
        torch.manual_seed(0)
        x = torch.rand(M, K, device="cuda")
        lim = 1.0 / (K ** 0.5)
        W = (torch.rand(N, K, device="cuda") * 2 - 1) * lim
        b = (torch.rand(N, device="cuda") * 2 - 1) * lim
        return dict(x=x, W=W, b=b)
    def torch_fn(ins):
        y = ins["x"] @ ins["W"].T + ins["b"]
        y = torch.clamp(y, max=constant)
        return y - constant
    def cuda_run(mod, ins):
        return mod.run(ins["x"], ins["W"], ins["b"], constant)
    return dict(make_inputs=make_inputs, torch_fn=torch_fn, cuda_run=cuda_run,
                in_shape=[M, K, N], out_shape=[M, N], atol=5e-3)


PROBLEMS = {
    "gemm_mul_lrelu": _gemm_mul_lrelu(),
    "conv_relu_hardswish": _conv_relu_hardswish(),
    "matmul_mish_mish": _matmul_mish_mish(),
    "matmul_scale_resadd": _matmul_scale_resadd(),
    "gemm_scale_htanh_gelu": _gemm_scale_htanh_gelu(),
    "matmul_sigmoid_sum": _matmul_sigmoid_sum(),
    "matmul_swish_scaling": _matmul_swish_scaling(),
    "gemm_relu_div": _gemm_relu_div(),
    "conv_relu_biasadd": _conv_relu_biasadd(),
    "matmul_sub_mul_relu": _matmul_sub_mul_relu(),
    "gemm_add_relu": _gemm_add_relu(),
    "matmul_div_gelu": _matmul_div_gelu(),
    "matmul_min_subtract": _matmul_min_subtract(),
}


def run_one(name):
    p = PROBLEMS[name]
    torch.cuda.empty_cache()
    ins = p["make_inputs"]()
    ref = p["torch_fn"](ins)
    torch_us = time_fn(lambda: p["torch_fn"](ins))

    # SeGuRu←PyTorch arm
    sg_us, sg_err, sg_error = float("inf"), float("inf"), ""
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td); (tmp / "in").mkdir(); (tmp / "out").mkdir()
            for k, v in ins.items(): dump_bin(tmp / f"in/{k}.bin", v)
            sg = run_seguru(name, tmp / "in", tmp / "out", iters=50, shape=p["in_shape"])
            y_sg = torch.from_numpy(load_bin(tmp / "out/y.bin", tuple(p["out_shape"]))).cuda()
            sg_err = float((y_sg - ref).abs().max())
            sg_us = sg["kernel_us"]
    except Exception as exc:
        sg_error = f"{type(exc).__name__}: {str(exc)[:120]}"

    # SeGuRu←CUDA arm
    fc_us, fc_err, fc_error = float("inf"), float("inf"), ""
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td); (tmp / "in").mkdir(); (tmp / "out").mkdir()
            for k, v in ins.items(): dump_bin(tmp / f"in/{k}.bin", v)
            fc = run_seguru(f"{name}_fc", tmp / "in", tmp / "out", iters=50, shape=p["in_shape"])
            y_fc = torch.from_numpy(load_bin(tmp / "out/y.bin", tuple(p["out_shape"]))).cuda()
            fc_err = float((y_fc - ref).abs().max())
            fc_us = fc["kernel_us"]
    except Exception as exc:
        fc_error = f"{type(exc).__name__}: {str(exc)[:120]}"

    # raw CUDA arm
    cu_us, cu_err, cu_error = float("inf"), float("inf"), ""
    mod, err = compile_cuda(name)
    if mod is None:
        cu_error = err
    else:
        try:
            y_cu = p["cuda_run"](mod, ins)
            cu_err = float((y_cu - ref).abs().max())
            cu_us = time_fn(lambda: p["cuda_run"](mod, ins))
        except Exception as exc:
            cu_error = f"{type(exc).__name__}: {str(exc)[:120]}"

    return dict(
        problem=name, atol=p["atol"],
        torch_us=torch_us,
        sg_us=sg_us, sg_err=sg_err, sg_error=sg_error,
        fc_us=fc_us, fc_err=fc_err, fc_error=fc_error,
        cu_us=cu_us, cu_err=cu_err, cu_error=cu_error,
    )


def fmt(r):
    def arm(us, err, atol, e=""):
        if us == float("inf"): return f"  FAILED ({e[:60]})"
        ok = "✓" if err <= max(atol, 5e-3) else "✗"
        return f"{us:9.1f}us {ok}"
    atol = r["atol"]
    return (f"  {r['problem']:22s}  "
            f"torch={r['torch_us']:9.1f}us  "
            f"seguru={arm(r['sg_us'], r['sg_err'], atol, r['sg_error'])}  "
            f"cuda={arm(r['cu_us'], r['cu_err'], atol, r['cu_error'])}  "
            f"seguru_fc={arm(r['fc_us'], r['fc_err'], atol, r['fc_error'])}")


def main():
    assert RUNNER.exists(), "build first: cargo build --release -p kernelbench-c"
    problems = sys.argv[1:] or list(PROBLEMS.keys())
    results = []
    for name in problems:
        print(f"... {name}", flush=True)
        r = run_one(name)
        print(fmt(r), flush=True)
        results.append(r)

    print("\n=== summary (correctness + speedup vs torch-eager) ===")
    for arm_us, arm_err, label in [("sg_us","sg_err","seguru"),
                                   ("cu_us","cu_err","cuda"),
                                   ("fc_us","fc_err","seguru_fc")]:
        n = len(results)
        ok = [r for r in results if r[arm_err] <= max(r["atol"], 5e-3) and r[arm_us] != float("inf")]
        fast1 = [r for r in ok if r["torch_us"]/r[arm_us] >= 1.0]
        fast2 = [r for r in ok if r["torch_us"]/r[arm_us] >= 2.0]
        avg = sum(r["torch_us"]/r[arm_us] for r in ok) / max(len(ok), 1)
        print(f"  {label:8s}  correct={len(ok)}/{n}  fast_1={len(fast1)*100//n:3d}%  "
              f"fast_2={len(fast2)*100//n:3d}%  avg_speedup={avg:.2f}x")


if __name__ == "__main__":
    main()
