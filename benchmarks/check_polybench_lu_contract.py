#!/usr/bin/env python3
"""Guard the PolyBench LU SeGuRu benchmark dataflow contract."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unmatched brace")


def block_after_marker(text: str, marker: str) -> tuple[int, int]:
    marker_index = text.index(marker)
    open_index = text.index("{", marker_index)
    close_index = find_matching_brace(text, open_index)
    return open_index + 1, close_index


COPY_TO_HOST = re.compile(
    r"\b(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\.\s*copy_to_host\s*\(\s*&\s*mut\s*(?P<buffer>[A-Za-z_][A-Za-z0-9_]*)\b"
)
COPY_FROM_HOST = re.compile(
    r"\b(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\.\s*copy_from_host\s*\(\s*&\s*(?P<buffer>[A-Za-z_][A-Za-z0-9_]*)\b"
)


def host_roundtrip(body: str) -> re.Match[str] | None:
    for copy_out in COPY_TO_HOST.finditer(body):
        for copy_in in COPY_FROM_HOST.finditer(body, copy_out.end()):
            if copy_in.group("buffer") == copy_out.group("buffer"):
                return copy_out
    return None


def check_lu_file(
    path: str,
    marker: str,
    declaration_tokens: tuple[str, ...],
    body_tokens: tuple[str, ...],
) -> list[str]:
    text = read(path)
    start, end = block_after_marker(text, marker)
    body = text[start:end]
    errors: list[str] = []

    roundtrip = host_roundtrip(body)
    if roundtrip is not None:
        errors.append(
            f"{path}:{line_number(text, start + roundtrip.start())}: "
            "LU still round-trips the full matrix through host memory; keep "
            "synchronization on device."
        )

    for token in declaration_tokens:
        if token not in text:
            errors.append(f"{path}: missing LU declaration token {token!r}")

    for token in body_tokens:
        if token not in body:
            errors.append(f"{path}: LU execution body missing token {token!r}")

    return errors


def main() -> int:
    errors = []
    errors.extend(
        check_lu_file(
            "examples/polybench/lu/src/lib.rs",
            "// GPU",
            (
                "pub fn lu_kernel1(pivot: &[f32], row_tail: &mut [f32], rem: u32)",
                "pub fn lu_copy_col",
                "pub fn lu_kernel2(row_tail: &[f32], col: &[f32], rows_below: &mut [f32], n: u32, k: u32)",
            ),
            (
                "let (prefix, mut tail_and_after) = d_a.split_at_mut(row_tail_start)",
                "lu_kernel1::launch",
                "&mut row_tail",
                "lu_copy_col::launch",
                "&mut d_col",
                "let (prefix, mut rows_below) = d_a.split_at_mut(split_at)",
                "let col = d_col.index(..rem)",
                "lu_kernel2::launch",
                "&mut rows_below",
            ),
        )
    )
    errors.extend(
        check_lu_file(
            "examples/bench-polybench/src/main.rs",
            "// --- lu (N=2048) ---",
            (
                "pub fn bench_lu_kernel1(pivot: &[f32], row_tail: &mut [f32], rem: u32)",
                "pub fn bench_lu_copy_col",
                "pub fn bench_lu_kernel2(row_tail: &[f32], col: &[f32], rows_below: &mut [f32], n: u32, k: u32)",
            ),
            (
                "let (prefix, mut tail_and_after) = d_a.split_at_mut(row_tail_start)",
                "bench_lu_kernel1::launch",
                "&mut row_tail",
                "bench_lu_copy_col::launch",
                "&mut d_col",
                "let (prefix, mut rows_below) = d_a.split_at_mut(split_at)",
                "let col = d_col.index(..rem)",
                "bench_lu_kernel2::launch",
                "&mut rows_below",
            ),
        )
    )

    if errors:
        print("\n".join(errors))
        return 1

    print("PolyBench LU uses device-side synchronization without full host round-trips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
