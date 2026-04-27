#!/usr/bin/env python3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED = {
    "examples/kernelbench-b/cuda/masked_cumsum.cu": [
        "masked_cumsum_kernel",
        "CUDAGuard",
        "getCurrentCUDAStream",
        "requires non-empty",
        "std::numeric_limits<int>::max()",
        "D % BLOCK == 0",
        "mask.device() == x.device()",
    ],
    "examples/kernelbench-b/src/masked_cumsum.rs": [
        "masked_cumsum_kernel",
        "checked_mul",
        "u32::try_from",
        "masked_cumsum requires non-empty",
        "copy_to_host",
    ],
    "examples/kernelbench-b/src/from_cuda/masked_cumsum.rs": [
        "masked_cumsum_kernel",
        "checked_mul",
        "u32::try_from",
        "masked_cumsum requires non-empty",
        "copy_to_host",
    ],
}


def main() -> None:
    for rel, tokens in REQUIRED.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        if missing:
            raise AssertionError(f"{rel} missing {missing}")
    print("KernelBench-B masked_cumsum optimization guard passed.")


if __name__ == "__main__":
    main()
