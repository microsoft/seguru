#!/usr/bin/env python3
"""Guard against regressing refreshed KernelBench-B ports back to stale scalar code."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def require(path: str, *tokens: str) -> None:
    text = (REPO / path).read_text()
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{path} missing tokens: {missing}")


def main() -> None:
    refreshed_fc = {
        "examples/kernelbench-b/src/from_cuda/mse_loss.rs": [
            "use gpu::vector::Float4;",
            "pub fn mse_loss_kernel_vec",
            "chunks_exact(4)",
            "mse_loss_kernel_vec::launch",
        ],
        "examples/kernelbench-b/src/from_cuda/max_dim.rs": [
            "use gpu::vector::Float4;",
            "pub fn max_dim_kernel_vec",
            "chunks_exact(4)",
            "max_dim_kernel_vec::launch",
        ],
        "examples/kernelbench-b/src/from_cuda/argmax_dim.rs": [
            "use gpu::vector::Float4;",
            "fn warp_min_i32",
            "pub fn argmax_dim_kernel_vec",
            "chunks_exact(4)",
            "argmax_dim_kernel_vec::launch",
        ],
        "examples/kernelbench-b/src/from_cuda/softmax.rs": [
            "use gpu::vector::{Float4, VecFlatten};",
            "pub fn softmax_kernel",
            "Float4::new([0.0; 4])",
            "chunks_exact(4)",
        ],
        "examples/kernelbench-b/src/from_cuda/log_softmax.rs": [
            "use gpu::vector::{Float4, VecFlatten};",
            "pub fn log_softmax_kernel",
            "Float4::new([0.0; 4])",
            "chunks_exact(4)",
        ],
        "examples/kernelbench-b/src/from_cuda/swish.rs": [
            "use gpu::vector::{Float4, VecFlatten};",
            "pub fn swish_kernel_vec",
            "chunks_exact(4)",
            "swish_kernel_vec::launch",
        ],
        "examples/kernelbench-b/src/from_cuda/softplus.rs": [
            "use gpu::vector::{Float4, VecFlatten};",
            "pub fn softplus_kernel_vec",
            "chunks_exact(4)",
            "softplus_kernel_vec::launch",
        ],
    }
    for path, tokens in refreshed_fc.items():
        require(path, *tokens)

    require(
        "examples/kernelbench-b/src/mse_loss.rs",
        "use gpu::vector::Float4;",
        "pub fn mse_loss_kernel_vec",
        "chunks_exact(4)",
        "mse_loss_kernel_vec::launch",
    )

    require(
        "examples/kernelbench-b/src/from_cuda/tanh.rs",
        "try_cast_slice::<Float4>()",
        "try_cast_slice_mut::<Float4>()",
    )
    require(
        "examples/kernelbench-b/src/from_cuda/sigmoid.rs",
        "try_cast_slice::<Float4>()",
        "try_cast_slice_mut::<Float4>()",
    )

    require(
        "examples/kernelbench-b/python/compare2.py",
        '"avg_pool1d": dict(',
        '"min_dim": dict(',
        'os.environ.get("CARGO_TARGET_DIR"',
    )

    print("KernelBench-B stale refresh guard passed.")


if __name__ == "__main__":
    main()
