#!/usr/bin/env python3
"""Source guard for the next KernelBench-B Level-1 batch."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

PROBLEMS = [
    "argmin_dim",
    "cumprod",
    "cumsum_reverse",
    "cumsum_exclusive",
    "masked_cumsum",
    "batch_norm",
    "instance_norm",
    "group_norm",
    "depthwise_conv2d",
    "pointwise_conv2d",
]


def read(path: str) -> str:
    full = REPO / path
    if not full.exists():
        raise AssertionError(f"missing file: {path}")
    return full.read_text()


def require(path: str, *tokens: str) -> None:
    text = read(path)
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{path} missing tokens: {missing}")


def require_order(path: str, first: str, second: str) -> None:
    text = read(path)
    first_pos = text.find(first)
    second_pos = text.find(second)
    if first_pos == -1 or second_pos == -1:
        raise AssertionError(f"{path} missing order tokens: {[first, second]}")
    if first_pos > second_pos:
        raise AssertionError(f"{path}: expected {first!r} before {second!r}")


def main() -> None:
    main_rs = read("examples/kernelbench-b/src/main.rs")
    fc_mod = read("examples/kernelbench-b/src/from_cuda/mod.rs")
    compare = read("examples/kernelbench-b/python/compare2.py")

    for problem in PROBLEMS:
        read(f"examples/kernelbench-b/cuda/{problem}.cu")
        read(f"examples/kernelbench-b/src/{problem}.rs")
        read(f"examples/kernelbench-b/src/from_cuda/{problem}.rs")

        if f"pub mod {problem};" not in main_rs:
            raise AssertionError(f"main.rs missing direct module {problem}")
        if f'"{problem}" =>' not in main_rs:
            raise AssertionError(f"main.rs missing direct dispatch {problem}")
        if f'"{problem}_fc" =>' not in main_rs:
            raise AssertionError(f"main.rs missing from-CUDA dispatch {problem}_fc")
        if f"pub mod {problem};" not in fc_mod:
            raise AssertionError(f"from_cuda/mod.rs missing module {problem}")
        if f'"{problem}": dict(' not in compare:
            raise AssertionError(f"compare2.py missing problem {problem}")

    require(
        "examples/kernelbench-b/python/compare2.py",
        "out_dtype=\"int64\"",
        'inputs=["x", "mask"]',
        'inputs=["x", "w"]',
        "F.batch_norm",
        "F.instance_norm",
        "F.group_norm",
        "F.conv2d",
    )
    require(
        "examples/kernelbench-b/src/argmin_dim.rs",
        "write_bin_i64",
    )
    require(
        "examples/kernelbench-b/src/masked_cumsum.rs",
        'read_bin(&in_dir.join("mask.bin")',
    )
    require(
        "examples/kernelbench-b/src/depthwise_conv2d.rs",
        'read_bin(&in_dir.join("w.bin")',
    )
    require(
        "examples/kernelbench-b/src/pointwise_conv2d.rs",
        'read_bin(&in_dir.join("w.bin")',
    )
    for path in [
        "examples/kernelbench-b/src/depthwise_conv2d.rs",
        "examples/kernelbench-b/src/pointwise_conv2d.rs",
        "examples/kernelbench-b/src/from_cuda/depthwise_conv2d.rs",
        "examples/kernelbench-b/src/from_cuda/pointwise_conv2d.rs",
    ]:
        require(path, "checked_mul", "u32::try_from")
    for path in [
        "examples/kernelbench-b/src/pointwise_conv2d.rs",
        "examples/kernelbench-b/src/from_cuda/pointwise_conv2d.rs",
    ]:
        require_order(
            path,
            'let total = checked_u32("pointwise_conv2d output elements", y_len);',
            "let h_x = crate::read_bin",
        )

    print("KernelBench-B next10 source guard passed.")


if __name__ == "__main__":
    main()
