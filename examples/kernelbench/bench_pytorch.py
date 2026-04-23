#!/usr/bin/env python3.11
"""PyTorch benchmark for KernelBench Level 1 kernels."""

import sys
import os
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.11/site-packages"))

import torch
import torch.nn.functional as F

# Configuration
WARMUP = 3
ITERS = 10

def bench_op(op_fn, warmup=WARMUP, iters=ITERS):
    """Run op_fn() with warmup, then time ITERS iterations, return median µs."""
    # Warmup
    for _ in range(warmup):
        op_fn()
        torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        op_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms → µs
    
    times.sort()
    return times[len(times) // 2]  # median

def gen_input_fast(n, device='cuda'):
    """Generate deterministic input matching Rust side: (i % 7) * 0.1"""
    idx = torch.arange(n, device=device, dtype=torch.float32)
    return (idx % 7) * 0.1

def gen_targets(n, device='cuda'):
    """Generate targets for loss functions: (i % 11) * 0.1"""
    idx = torch.arange(n, device=device, dtype=torch.float32)
    return (idx % 11) * 0.1

def rms_norm(x, eps=1e-8):
    """RMS normalization: x / sqrt(mean(x²) + eps)"""
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)

def hinge_loss(pred, target):
    """Hinge loss: mean(max(0, 1 - pred * target))"""
    return torch.mean(torch.max(torch.zeros_like(pred), 1 - pred * target))

def cumsum_exclusive(x):
    """Exclusive cumulative sum: shift inclusive cumsum right by 1"""
    inc = torch.cumsum(x, dim=-1)
    exc = torch.zeros_like(x)
    exc[:, 1:] = inc[:, :-1]
    return exc

def cumsum_reverse(x):
    """Reverse cumulative sum"""
    return torch.flip(torch.cumsum(torch.flip(x, [-1]), dim=-1), [-1])

# Kernel operation mappings
OPERATIONS = {
    # Elementwise (11)
    "relu_forward": lambda x: torch.relu(x),
    "leaky_relu_forward": lambda x: F.leaky_relu(x, 0.01),
    "sigmoid_forward": lambda x: torch.sigmoid(x),
    "tanh_forward": lambda x: torch.tanh(x),
    "swish_forward": lambda x: F.silu(x),  # SiLU = Swish
    "selu_forward": lambda x: F.selu(x),
    "hard_sigmoid_forward": lambda x: F.hardsigmoid(x),
    "softplus_forward": lambda x: F.softplus(x),
    "softsign_forward": lambda x: F.softsign(x),
    "elu_forward": lambda x: F.elu(x, alpha=1.0),
    "hard_tanh_forward": lambda x: F.hardtanh(x, -1.0, 1.0),

    # GELU (2)
    "gelu_forward": lambda x: F.gelu(x),
    "mingpt_new_gelu_forward": lambda x: F.gelu(x, approximate="tanh"),

    # Matmul 2D (4)
    "matmul_forward": lambda a, b: torch.matmul(a, b),
    "matmul_transposed_a": lambda a, b: torch.matmul(a.t(), b),
    "matmul_transposed_b": lambda a, b: torch.matmul(a, b.t()),
    "matmul_transposed_both": lambda a, b: torch.matmul(a.t(), b.t()),

    # Batched matmul (2)
    "matmul_batched": lambda a, b: torch.bmm(a, b),
    "tensor3d_matmul": lambda a, b: torch.bmm(a, b),

    # Matvec (3)
    "matvec_forward": lambda a, x: torch.mv(a, x),
    "scalar_multiply": lambda x, s: x * s,
    "tensor3d_matmul_matvec": lambda a, b: torch.bmm(a, b),

    # Reduction (4)
    "sum_reduce": lambda x: torch.sum(x, dim=-1),
    "mean_reduce": lambda x: torch.mean(x, dim=-1),
    "max_reduce": lambda x: torch.max(x, dim=-1).values,
    "min_reduce": lambda x: torch.min(x, dim=-1).values,

    # Argreduce (2)
    "argmax_reduce": lambda x: torch.argmax(x, dim=-1),
    "argmin_reduce": lambda x: torch.argmin(x, dim=-1),

    # Softmax (2)
    "softmax_forward": lambda x: F.softmax(x, dim=-1),
    "log_softmax_forward": lambda x: F.log_softmax(x, dim=-1),

    # Norm (5)
    "rms_norm_forward": lambda x: rms_norm(x),
    "frobenius_norm_forward": lambda x: torch.norm(x),
    "l1_norm_forward": lambda x: x / torch.sum(torch.abs(x), dim=-1, keepdim=True),
    "l2_norm_forward": lambda x: F.normalize(x, p=2, dim=-1),
    "layer_norm_forward": lambda x: F.layer_norm(x, [x.shape[-1]]),

    # Loss (4)
    "mse_loss_forward": lambda p, t: F.mse_loss(p, t),
    "huber_loss_forward": lambda p, t: F.huber_loss(p, t, delta=1.0),
    "kl_div_loss_forward": lambda logp, t: F.kl_div(logp, t, reduction='batchmean'),
    "hinge_loss_forward": lambda p, t: hinge_loss(p, t),

    # Cumulative (4)
    "cumsum_forward": lambda x: torch.cumsum(x, dim=-1),
    "cumprod_forward": lambda x: torch.cumprod(x, dim=-1),
    "cumsum_reverse_forward": lambda x: cumsum_reverse(x),
    "cumsum_exclusive_forward": lambda x: cumsum_exclusive(x),
}

# Categories and their size specifications
CATEGORIES = {
    "elementwise": {
        "kernels": ["relu_forward", "leaky_relu_forward", "sigmoid_forward", "tanh_forward", 
                   "swish_forward", "selu_forward", "hard_sigmoid_forward", "softplus_forward",
                   "softsign_forward", "elu_forward", "hard_tanh_forward"],
        "small": {"n": 4096},
        "large": {"n": 1_048_576}
    },
    "gelu": {
        "kernels": ["gelu_forward", "mingpt_new_gelu_forward"],
        "small": {"n": 4096},
        "large": {"n": 1_048_576}
    },
    "matmul2d": {
        "kernels": ["matmul_forward", "matmul_transposed_a", "matmul_transposed_b", "matmul_transposed_both"],
        "small": {"M": 64, "N": 64, "K": 64},
        "large": {"M": 1024, "N": 1024, "K": 1024}
    },
    "batched_matmul": {
        "kernels": ["matmul_batched", "tensor3d_matmul"],
        "small": {"B": 4, "M": 32, "N": 32, "K": 32},
        "large": {"B": 16, "M": 256, "N": 256, "K": 256}
    },
    "matvec": {
        "kernels": ["matvec_forward"],
        "small": {"M": 64, "N": 64},
        "large": {"M": 4096, "N": 4096}
    },
    "scalar": {
        "kernels": ["scalar_multiply"],
        "small": {"n": 4096},
        "large": {"n": 1_048_576}
    },
    "tensor3d": {
        "kernels": ["tensor3d_matmul_matvec"],
        "small": {"B": 4, "M": 32, "N": 32, "K": 32},
        "large": {"B": 16, "M": 256, "N": 256, "K": 256}
    },
    "reduction": {
        "kernels": ["sum_reduce", "mean_reduce", "max_reduce", "min_reduce"],
        "small": {"batch": 64, "dim": 256},
        "large": {"batch": 1024, "dim": 4096}
    },
    "argreduce": {
        "kernels": ["argmax_reduce", "argmin_reduce"],
        "small": {"batch": 64, "dim": 256},
        "large": {"batch": 1024, "dim": 4096}
    },
    "softmax": {
        "kernels": ["softmax_forward", "log_softmax_forward"],
        "small": {"batch": 64, "dim": 256},
        "large": {"batch": 1024, "dim": 4096}
    },
    "norm": {
        "kernels": ["rms_norm_forward", "l1_norm_forward", "l2_norm_forward", "layer_norm_forward"],
        "small": {"batch": 64, "dim": 256},
        "large": {"batch": 1024, "dim": 4096}
    },
    "norm_global": {
        "kernels": ["frobenius_norm_forward"],
        "small": {"n": 4096},
        "large": {"n": 1_048_576}
    },
    "loss": {
        "kernels": ["mse_loss_forward", "huber_loss_forward", "kl_div_loss_forward", "hinge_loss_forward"],
        "small": {"n": 4096},
        "large": {"n": 1_048_576}
    },
    "cumulative": {
        "kernels": ["cumsum_forward", "cumprod_forward", "cumsum_reverse_forward", "cumsum_exclusive_forward"],
        "small": {"batch": 64, "dim": 256},
        "large": {"batch": 1024, "dim": 4096}
    }
}

def benchmark_kernel(kernel_name, category, size_config, size_label):
    """Benchmark a single kernel at given size."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    op = OPERATIONS[kernel_name]
    
    try:
        # Generate inputs based on category
        if category in ["elementwise", "gelu", "scalar", "norm_global", "loss"]:
            n = size_config["n"]
            if kernel_name == "scalar_multiply":
                x = gen_input_fast(n, device)
                s = torch.tensor(2.5, device=device)
                op_fn = lambda: op(x, s)
            elif kernel_name in ["mse_loss_forward", "huber_loss_forward", "hinge_loss_forward"]:
                pred = gen_input_fast(n, device)
                target = gen_targets(n, device)
                op_fn = lambda: op(pred, target)
            elif kernel_name == "kl_div_loss_forward":
                pred = gen_input_fast(n, device) + 1e-8  # Avoid log(0)
                log_pred = torch.log(pred)
                target = gen_targets(n, device) + 1e-8
                target = target / torch.sum(target)  # Normalize
                op_fn = lambda: op(log_pred, target)
            else:
                x = gen_input_fast(n, device)
                op_fn = lambda: op(x)
            n_elements = n
            
        elif category == "matmul2d":
            M, N, K = size_config["M"], size_config["N"], size_config["K"]
            if kernel_name == "matmul_forward":
                a = gen_input_fast(M * K, device).reshape(M, K)
                b = gen_input_fast(K * N, device).reshape(K, N)
            elif kernel_name == "matmul_transposed_a":
                a = gen_input_fast(K * M, device).reshape(K, M)  # Stored as K×M
                b = gen_input_fast(K * N, device).reshape(K, N)
            elif kernel_name == "matmul_transposed_b":
                a = gen_input_fast(M * K, device).reshape(M, K)
                b = gen_input_fast(N * K, device).reshape(N, K)  # Stored as N×K
            elif kernel_name == "matmul_transposed_both":
                a = gen_input_fast(K * M, device).reshape(K, M)  # Stored as K×M
                b = gen_input_fast(N * K, device).reshape(N, K)  # Stored as N×K
            op_fn = lambda: op(a, b)
            n_elements = M * N
            
        elif category == "batched_matmul" or category == "tensor3d":
            B, M, N, K = size_config["B"], size_config["M"], size_config["N"], size_config["K"]
            a = gen_input_fast(B * M * K, device).reshape(B, M, K)
            b = gen_input_fast(B * K * N, device).reshape(B, K, N)
            op_fn = lambda: op(a, b)
            n_elements = B * M * N
            
        elif category == "matvec":
            M, N = size_config["M"], size_config["N"]
            a = gen_input_fast(M * N, device).reshape(M, N)
            x = gen_input_fast(N, device)
            op_fn = lambda: op(a, x)
            n_elements = M
            
        elif category in ["reduction", "argreduce", "softmax", "norm", "cumulative"]:
            batch, dim = size_config["batch"], size_config["dim"]
            x = gen_input_fast(batch * dim, device).reshape(batch, dim)
            op_fn = lambda: op(x)
            n_elements = batch * dim
            
        # Benchmark
        time_us = bench_op(op_fn)
        return {
            "kernel": kernel_name,
            "category": category,
            "size_label": size_label,
            "n_elements": n_elements,
            "pytorch_us": time_us
        }
        
    except Exception as e:
        print(f"Error benchmarking {kernel_name}: {e}", file=sys.stderr)
        return None

def main():
    """Main benchmarking function."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU timing", file=sys.stderr)
        # Modify bench_op for CPU
        global bench_op
        def cpu_bench_op(op_fn, warmup=WARMUP, iters=ITERS):
            import time
            # Warmup
            for _ in range(warmup):
                op_fn()
            
            # Timing
            times = []
            for _ in range(iters):
                start = time.perf_counter()
                op_fn()
                end = time.perf_counter()
                times.append((end - start) * 1_000_000)  # s → µs
            
            times.sort()
            return times[len(times) // 2]  # median
        bench_op = cpu_bench_op
    
    results = []
    
    print("kernel,category,size_label,n_elements,pytorch_us", file=sys.stdout)
    
    # Benchmark all kernels
    total_kernels = sum(len(cat_info["kernels"]) for cat_info in CATEGORIES.values())
    kernel_count = 0
    
    for category, cat_info in CATEGORIES.items():
        for kernel_name in cat_info["kernels"]:
            kernel_count += 1
            print(f"Benchmarking {kernel_name} ({kernel_count}/{total_kernels})...", file=sys.stderr)
            
            # Small size
            result = benchmark_kernel(kernel_name, category, cat_info["small"], "small")
            if result:
                results.append(result)
                print(f"{result['kernel']},{result['category']},{result['size_label']},{result['n_elements']},{result['pytorch_us']:.2f}")
            
            # Large size
            result = benchmark_kernel(kernel_name, category, cat_info["large"], "large")
            if result:
                results.append(result)
                print(f"{result['kernel']},{result['category']},{result['size_label']},{result['n_elements']},{result['pytorch_us']:.2f}")
    
    # Summary
    print(f"\n=== PyTorch Benchmark Summary ===", file=sys.stderr)
    print(f"Total kernels benchmarked: {len(results) // 2}", file=sys.stderr)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", file=sys.stderr)

if __name__ == "__main__":
    main()