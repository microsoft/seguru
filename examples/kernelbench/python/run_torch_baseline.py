"""PyTorch baseline timings for the KernelBench L1 subset (matches shapes used
by examples/kernelbench/src/main.rs). Uses CUDA events for GPU timing.

Run:  python examples/kernelbench/python/run_torch_baseline.py
"""
import torch

ITERS = 100
WARMUP = 10

def bench(name, shape, fn, *args):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / ITERS  # ms -> us
    print(f"{name:10s} {str(shape):20s} PyTorch: {us:8.2f} us")

def main():
    dev = 'cuda'
    # ReLU / Sigmoid
    x = torch.randn(4096, 16384, device=dev)
    bench('relu',    (4096, 16384), torch.relu, x)
    bench('sigmoid', (4096, 16384), torch.sigmoid, x)

    # Softmax row-wise
    xs = torch.randn(4096, 4096, device=dev)
    bench('softmax', (4096, 4096), lambda z: torch.softmax(z, dim=1), xs)

    # Matmul
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    bench('matmul',  (4096, 4096), torch.matmul, a, b)

if __name__ == '__main__':
    main()
