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
        "input byte count overflow",
        "u32::try_from",
        "masked_cumsum requires non-empty",
        "copy_to_host",
    ],
    "examples/kernelbench-b/src/from_cuda/masked_cumsum.rs": [
        "masked_cumsum_kernel",
        "checked_mul",
        "input byte count overflow",
        "u32::try_from",
        "masked_cumsum requires non-empty",
        "copy_to_host",
    ],
}

ORDERED = {
    "examples/kernelbench-b/src/masked_cumsum.rs": [
        ("u32::try_from(d)", "read_bin"),
        ("u32::try_from(b)", "read_bin"),
        ("copy_to_host", "write_bin"),
    ],
    "examples/kernelbench-b/src/from_cuda/masked_cumsum.rs": [
        ("u32::try_from(d)", "read_bin"),
        ("u32::try_from(b)", "read_bin"),
        ("copy_to_host", "write_bin"),
    ],
}


def main() -> None:
    failures = []
    for rel, tokens in REQUIRED.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        if missing:
            failures.append(f"{rel} missing {missing}")

    for rel, pairs in ORDERED.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        for before, after in pairs:
            before_idx = text.find(before)
            after_idx = text.find(after)
            if before_idx == -1 or after_idx == -1:
                continue
            if before_idx > after_idx:
                failures.append(f"{rel} must place {before!r} before {after!r}")

    if failures:
        raise AssertionError("\n".join(failures))

    print("KernelBench-B masked_cumsum optimization guard passed.")


if __name__ == "__main__":
    main()
