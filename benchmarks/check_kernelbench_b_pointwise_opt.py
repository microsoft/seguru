#!/usr/bin/env python3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED = {
    "examples/kernelbench-b/cuda/pointwise_conv2d.cu": [
        "TILE_M",
        "TILE_N",
        "TILE_K",
        "__shared__",
        "pointwise_conv2d_tiled_kernel",
    ],
    "examples/kernelbench-b/src/pointwise_conv2d.rs": [
        "TILE_M",
        "TILE_N",
        "TILE_K",
        "GpuShared",
        "pointwise_conv2d_tiled_kernel",
    ],
    "examples/kernelbench-b/src/from_cuda/pointwise_conv2d.rs": [
        "TILE_M",
        "TILE_N",
        "TILE_K",
        "GpuShared",
        "pointwise_conv2d_tiled_kernel",
    ],
}

for rel, tokens in REQUIRED.items():
    text = (REPO / rel).read_text()
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{rel} missing {missing}")

print("KernelBench-B pointwise conv optimization guard passed.")
