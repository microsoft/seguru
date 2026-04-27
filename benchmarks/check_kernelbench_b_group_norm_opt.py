#!/usr/bin/env python3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED = {
    "examples/kernelbench-b/cuda/group_norm.cu": [
        "group_norm_stats_kernel",
        "group_norm_apply_kernel",
        "float4",
        "group_elems4",
        "rstd",
    ],
    "examples/kernelbench-b/src/group_norm.rs": [
        "group_norm_stats_kernel",
        "group_norm_apply_kernel",
        "Float4",
        "group_elems4",
        "chunk_to_scope(grid2block",
    ],
    "examples/kernelbench-b/src/from_cuda/group_norm.rs": [
        "group_norm_stats_kernel",
        "group_norm_apply_kernel",
        "Float4",
        "group_elems4",
        "chunk_to_scope(grid2block",
    ],
}


def main() -> None:
    for rel, tokens in REQUIRED.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        if missing:
            raise AssertionError(f"{rel} missing {missing}")
    print("KernelBench-B group_norm optimization guard passed.")


if __name__ == "__main__":
    main()
