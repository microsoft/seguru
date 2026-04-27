#!/usr/bin/env python3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED = {
    "examples/kernelbench-b/cuda/instance_norm.cu": [
        "instance_norm_stats_kernel",
        "instance_norm_apply_kernel",
        "instance_norm_stats_kernel<BLOCK><<<",
        "instance_norm_apply_kernel<<<",
        "float4",
        "hw4",
        "rstd",
        "CUDAGuard",
        "getCurrentCUDAStream",
        "requires non-empty",
        "std::numeric_limits<int>::max()",
        "std::numeric_limits<int>::max() - BLOCK",
    ],
    "examples/kernelbench-b/src/instance_norm.rs": [
        "instance_norm_stats_kernel",
        "instance_norm_apply_kernel",
        "instance_norm_stats_kernel::launch",
        "instance_norm_apply_kernel::launch",
        "Float4",
        "hw4",
        "rstd",
        "chunk_to_scope(grid2block",
        "mean_chunk",
        "rstd_chunk",
        "mean_chunk[0]",
        "rstd_chunk[0]",
        "checked_mul",
        "u32::try_from",
        "instance_norm requires non-empty",
    ],
    "examples/kernelbench-b/src/from_cuda/instance_norm.rs": [
        "instance_norm_stats_kernel",
        "instance_norm_apply_kernel",
        "instance_norm_stats_kernel::launch",
        "instance_norm_apply_kernel::launch",
        "Float4",
        "hw4",
        "rstd",
        "chunk_to_scope(grid2block",
        "mean_chunk",
        "rstd_chunk",
        "mean_chunk[0]",
        "rstd_chunk[0]",
        "checked_mul",
        "u32::try_from",
        "instance_norm requires non-empty",
    ],
}


def main() -> None:
    for rel, tokens in REQUIRED.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        if missing:
            raise AssertionError(f"{rel} missing {missing}")
    print("KernelBench-B instance_norm optimization guard passed.")


if __name__ == "__main__":
    main()
