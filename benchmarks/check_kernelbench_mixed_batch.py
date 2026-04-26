#!/usr/bin/env python3
"""Guard the planned mixed KernelBench L1/L2 batch source wiring."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "examples/kernelbench/src/main.rs": [
        "pub fn avg_pool1d_kernel",
        "pub fn min_dim1_kernel",
        '"avg_pool1d" =>',
        '"min_dim1" =>',
        "avg_pool1d_kernel::launch",
        "min_dim1_kernel::launch",
    ],
    "examples/kernelbench/python/driver.py": [
        "def problem_avg_pool1d",
        "def problem_min_dim1",
        "problem_avg_pool1d,",
        "problem_min_dim1,",
    ],
    "examples/kernelbench-c/python/compare.py": [
        "def _matmul_swish_scaling",
        "def _conv2d_scaling_min",
        '"matmul_swish_scaling": _matmul_swish_scaling()',
        '"conv2d_scaling_min": _conv2d_scaling_min()',
    ],
    "examples/kernelbench-c/src/main.rs": [
        "pub mod matmul_swish_scaling",
        "pub mod conv2d_scaling_min",
        '"matmul_swish_scaling"      => matmul_swish_scaling::run',
        '"conv2d_scaling_min"        => conv2d_scaling_min::run',
        '"matmul_swish_scaling_fc"   => from_cuda::matmul_swish_scaling::run',
        '"conv2d_scaling_min_fc"     => from_cuda::conv2d_scaling_min::run',
    ],
    "examples/kernelbench-c/src/from_cuda/mod.rs": [
        "pub mod matmul_swish_scaling",
        "pub mod conv2d_scaling_min",
    ],
}

REQUIRED_FILES = [
    "examples/kernelbench-c/problems/59_Matmul_Swish_Scaling.py",
    "examples/kernelbench-c/problems/32_Conv2d_Scaling_Min.py",
    "examples/kernelbench-c/cuda/matmul_swish_scaling.cu",
    "examples/kernelbench-c/cuda/conv2d_scaling_min.cu",
    "examples/kernelbench-c/src/matmul_swish_scaling.rs",
    "examples/kernelbench-c/src/from_cuda/matmul_swish_scaling.rs",
    "examples/kernelbench-c/src/conv2d_scaling_min.rs",
    "examples/kernelbench-c/src/from_cuda/conv2d_scaling_min.rs",
]


def check_required_tokens() -> list[str]:
    errors: list[str] = []
    for relative_path, tokens in REQUIRED.items():
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"{relative_path}: missing required source file")
            continue

        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"{relative_path}: missing token {token!r}")

    return errors


def check_required_files() -> list[str]:
    return [
        f"{relative_path}: missing required file"
        for relative_path in REQUIRED_FILES
        if not (ROOT / relative_path).is_file()
    ]


def main() -> int:
    errors = check_required_tokens() + check_required_files()
    if errors:
        print("\n".join(errors))
        return 1

    print("KernelBench mixed batch source wiring is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
