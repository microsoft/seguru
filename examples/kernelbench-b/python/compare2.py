#!/usr/bin/env python3.11
"""Phase B.N — 4-arm comparison (PyTorch eager | PyTorch compile | SeGuRu | raw CUDA).

Each arm runs the same problem on the same GPU tensor, timed with torch.cuda.Event.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, tempfile, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", REPO / "examples/target")) / "release/kernelbench-b"
CUDA_DIR = REPO / "examples/kernelbench-b/cuda"
BUILD_DIR = REPO / "examples/kernelbench-b/cuda_build"
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
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return mod, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:80]}"


# ---------- problem registry ----------
# Each entry: (torch_fn, make_x, in_shape, out_shape, cuda_run, atol)
# torch_fn and cuda_run take the same input shape and return a same-output-shape tensor.

def _rand(*s):
    return lambda: (torch.rand(*s, device="cuda") * 2 - 1).contiguous()

def _pos(*s):
    return lambda: torch.rand(*s, device="cuda").contiguous()


PROBLEMS = {
    "leaky_relu": dict(
        torch_fn=lambda x: F.leaky_relu(x, 0.01),
        make_x=_rand(2048, 65536),
        in_shape=[2048, 65536], out_shape=[2048, 65536],
        cuda_run=lambda mod, x: mod.run(x, 0.01),
        atol=1e-5,
    ),
    "tanh": dict(
        torch_fn=lambda x: torch.tanh(x),
        make_x=_rand(2048, 65536),
        in_shape=[2048, 65536], out_shape=[2048, 65536],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "rms_norm": dict(
        torch_fn=lambda x: x / torch.sqrt(torch.mean(x*x, dim=1, keepdim=True) + 1e-5),
        make_x=_pos(112, 64, 512, 512),
        in_shape=[112, 64, 512, 512], out_shape=[112, 64, 512, 512],
        cuda_run=lambda mod, x: mod.run(x, 1e-5),
        atol=1e-5,
    ),
    "relu": dict(
        torch_fn=lambda x: F.relu(x),
        make_x=_rand(2048, 65536),
        in_shape=[2048, 65536], out_shape=[2048, 65536],
        cuda_run=lambda mod, x: mod.run(x),
        atol=0,
    ),
    "sigmoid": dict(
        torch_fn=lambda x: torch.sigmoid(x),
        make_x=_rand(2048, 65536),
        in_shape=[2048, 65536], out_shape=[2048, 65536],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "gelu": dict(
        torch_fn=lambda x: F.gelu(x, approximate="tanh"),
        make_x=_rand(2048, 65536),
        in_shape=[2048, 65536], out_shape=[2048, 65536],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "softmax": dict(
        # softmax along last dim of a [B, D] matrix
        torch_fn=lambda x: torch.softmax(x, dim=-1),
        make_x=_rand(4096, 8192),
        in_shape=[4096, 8192], out_shape=[4096, 8192],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "layer_norm": dict(
        # LayerNorm over last dim, no affine, eps=1e-5
        torch_fn=lambda x: F.layer_norm(x, (x.shape[-1],), eps=1e-5),
        make_x=_rand(4096, 8192),
        in_shape=[4096, 8192], out_shape=[4096, 8192],
        cuda_run=lambda mod, x: mod.run(x, 1e-5),
        atol=1e-4,
    ),
    "sum_dim": dict(
        # sum over last dim: [B, D] -> [B]
        torch_fn=lambda x: torch.sum(x, dim=-1),
        make_x=_rand(4096, 16384),
        in_shape=[4096, 16384], out_shape=[4096],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-2,  # fp32 reduction of 16K elements
    ),
    "l2_norm": dict(
        # per-row L2 normalize: y[i] = x[i] / ||x[i]||_2 along last dim
        torch_fn=lambda x: x / torch.sqrt(torch.sum(x*x, dim=-1, keepdim=True) + 1e-12),
        make_x=_rand(4096, 8192),
        in_shape=[4096, 8192], out_shape=[4096, 8192],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "log_softmax": dict(
        torch_fn=lambda x: F.log_softmax(x, dim=-1),
        make_x=_rand(4096, 8192),
        in_shape=[4096, 8192], out_shape=[4096, 8192],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-4,
    ),
    "swish": dict(
        torch_fn=lambda x: x * torch.sigmoid(x),
        make_x=_rand(2048, 65536),
        in_shape=[2048, 65536], out_shape=[2048, 65536],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "softplus": dict(
        torch_fn=lambda x: F.softplus(x),
        make_x=_rand(2048, 65536),
        in_shape=[2048, 65536], out_shape=[2048, 65536],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "l1_norm": dict(
        torch_fn=lambda x: x / (x.abs().sum(dim=-1, keepdim=True) + 1e-12),
        make_x=_rand(4096, 8192),
        in_shape=[4096, 8192], out_shape=[4096, 8192],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "max_pool1d": dict(
        torch_fn=lambda x: F.max_pool1d(x, 4, 4),
        make_x=_rand(128, 64, 65536),
        in_shape=[128, 64, 65536], out_shape=[128, 64, 16384],
        cuda_run=lambda mod, x: mod.run(x),
        atol=0,
    ),
    "avg_pool1d": dict(
        torch_fn=lambda x: F.avg_pool1d(x, kernel_size=8, stride=1, padding=4),
        make_x=_pos(128, 64, 65536),
        in_shape=[128, 64, 65536], out_shape=[128, 64, 65537],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-5,
    ),
    "mean_dim": dict(
        torch_fn=lambda x: x.mean(dim=-1),
        make_x=_rand(4096, 16384),
        in_shape=[4096, 16384], out_shape=[4096],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-2,
    ),
    "max_dim": dict(
        torch_fn=lambda x: x.max(dim=-1).values,
        make_x=_rand(4096, 16384),
        in_shape=[4096, 16384], out_shape=[4096],
        cuda_run=lambda mod, x: mod.run(x),
        atol=0,
    ),
    "argmax_dim": dict(
        torch_fn=lambda x: x.argmax(dim=-1).to(torch.int64),
        make_x=_rand(4096, 16384),
        in_shape=[4096, 16384], out_shape=[4096],
        cuda_run=lambda mod, x: mod.run(x),
        atol=0,
        out_dtype="int64",
    ),
    "cumsum": dict(
        torch_fn=lambda x: torch.cumsum(x, dim=-1),
        make_x=_rand(1024, 4096),
        in_shape=[1024, 4096], out_shape=[1024, 4096],
        cuda_run=lambda mod, x: mod.run(x),
        atol=1e-3,
    ),
    "mse_loss": dict(
        torch_fn=lambda ab: F.mse_loss(ab[0], ab[1]),
        make_x=lambda: (
            (torch.rand(4096, 4096, device="cuda") * 2 - 1).contiguous(),
            (torch.rand(4096, 4096, device="cuda") * 2 - 1).contiguous(),
        ),
        in_shape=[4096, 4096], out_shape=[1],
        cuda_run=lambda mod, ab: mod.run(ab[0], ab[1]),
        atol=1e-3,
        inputs=["a", "b"],
    ),
}


def run_one(name):
    p = PROBLEMS[name]
    torch.cuda.empty_cache()
    x = p["make_x"]()
    ref = p["torch_fn"](x)
    torch_us = time_fn(lambda: p["torch_fn"](x))

    inputs = p.get("inputs", ["x"])
    out_dtype = p.get("out_dtype", "float32")
    out_np_dtype = np.int64 if out_dtype == "int64" else np.float32

    def dump_all(in_dir):
        if len(inputs) == 1:
            dump_bin(in_dir / f"{inputs[0]}.bin", x)
        else:
            for name_i, xi in zip(inputs, x):
                dump_bin(in_dir / f"{name_i}.bin", xi)

    def load_out(path, shape):
        arr = np.fromfile(path, dtype=out_np_dtype)
        if len(shape) > 0:
            arr = arr.reshape(shape)
        return arr

    def compare(y, ref_):
        if out_dtype == "int64":
            return float((y.long() - ref_.long()).abs().max())
        return float((y - ref_).abs().max())

    # torch.compile arm — DISABLED
    tc_us, tc_err, tc_error = float("inf"), float("inf"), "disabled"

    # SeGuRu arm
    sg_us, sg_err, sg_error = float("inf"), float("inf"), ""
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td); (tmp / "in").mkdir(); (tmp / "out").mkdir()
            dump_all(tmp / "in")
            sg = run_seguru(name, tmp / "in", tmp / "out", iters=50, shape=p["in_shape"])
            y_sg_np = load_out(tmp / "out/y.bin", tuple(p["out_shape"]))
            y_sg = torch.from_numpy(y_sg_np).cuda()
            sg_err = compare(y_sg, ref)
            sg_us = sg["kernel_us"]
    except Exception as exc:
        sg_error = f"{type(exc).__name__}: {str(exc)[:80]}"

    # SeGuRu-from-CUDA arm
    fc_us, fc_err, fc_error = float("inf"), float("inf"), ""
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td); (tmp / "in").mkdir(); (tmp / "out").mkdir()
            dump_all(tmp / "in")
            fc = run_seguru(f"{name}_fc", tmp / "in", tmp / "out", iters=50, shape=p["in_shape"])
            y_fc_np = load_out(tmp / "out/y.bin", tuple(p["out_shape"]))
            y_fc = torch.from_numpy(y_fc_np).cuda()
            fc_err = compare(y_fc, ref)
            fc_us = fc["kernel_us"]
    except Exception as exc:
        fc_error = f"{type(exc).__name__}: {str(exc)[:80]}"

    # CUDA arm
    cu_us, cu_err, cu_error = float("inf"), float("inf"), ""
    mod, err = compile_cuda(name)
    if mod is None:
        cu_error = err
    else:
        try:
            y_cu = p["cuda_run"](mod, x)
            cu_err = compare(y_cu, ref)
            cu_us = time_fn(lambda: p["cuda_run"](mod, x))
        except Exception as exc:
            cu_error = f"{type(exc).__name__}: {str(exc)[:80]}"

    return dict(
        problem=name, atol=p["atol"],
        torch_us=torch_us,
        tc_us=tc_us, tc_err=tc_err, tc_error=tc_error,
        sg_us=sg_us, sg_err=sg_err, sg_error=sg_error,
        fc_us=fc_us, fc_err=fc_err, fc_error=fc_error,
        cu_us=cu_us, cu_err=cu_err, cu_error=cu_error,
    )


def fmt(r):
    def arm(us, err, atol, e=""):
        if us == float("inf"): return f"  FAILED ({e[:40]})"
        ok = "✓" if err <= max(atol, 1e-3) else "✗"
        return f"{us:8.1f}us {ok}"
    atol = r["atol"]
    return (f"  {r['problem']:12s}  "
            f"torch={r['torch_us']:7.1f}us  "
            f"seguru={arm(r['sg_us'], r['sg_err'], atol, r['sg_error'])}  "
            f"cuda={arm(r['cu_us'], r['cu_err'], atol, r['cu_error'])}  "
            f"seguru_fc={arm(r['fc_us'], r['fc_err'], atol, r['fc_error'])}")


def main():
    assert RUNNER.exists(), "build first: cargo build --release -p kernelbench-b"
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
        ok = [r for r in results if r[arm_err] <= max(r["atol"], 1e-3) and r[arm_us] != float("inf")]
        fast1 = [r for r in ok if r["torch_us"]/r[arm_us] >= 1.0]
        fast2 = [r for r in ok if r["torch_us"]/r[arm_us] >= 2.0]
        avg = sum(r["torch_us"]/r[arm_us] for r in ok) / max(len(ok), 1)
        print(f"  {label:8s}  correct={len(ok)}/{n}  fast_1={len(fast1)*100//n:3d}%  "
              f"fast_2={len(fast2)*100//n:3d}%  avg_speedup={avg:.2f}x")


if __name__ == "__main__":
    main()
