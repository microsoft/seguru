#!/usr/bin/env python3
"""Guard the PolyBench GramSchmidt CUDA/SeGuRu transfer contract."""

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


def block_after_pattern(text: str, pattern: str, start: int = 0) -> tuple[int, int]:
    match = re.search(pattern, text[start:], re.MULTILINE)
    if match is None:
        raise ValueError(f"pattern not found: {pattern!r}")
    header_start = start + match.start()
    open_index = text.index("{", header_start)
    close_index = find_matching_brace(text, open_index)
    return open_index + 1, close_index


def require_tokens(text: str, path: str, tokens: tuple[str, ...]) -> list[str]:
    return [f"{path}: missing GramSchmidt token {token!r}" for token in tokens if token not in text]


def check_cuda() -> list[str]:
    path = "benchmarks/cuda/bench_gramschm.cu"
    text = read(path)
    errors = require_tokens(
        text,
        path,
        (
            "__global__ void gs_norm",
            "gs_norm<<<1,1>>>(d_a,d_r,NI,NJ,k)",
            "gs_normalize<<<(NI+255)/256,block1>>>(d_q,d_a,d_r,NI,NJ,k)",
        ),
    )
    scalar_copy = re.search(r"cudaMemcpy\s*\(\s*&\s*col\s*\[\s*i\s*\]", text)
    if scalar_copy is not None:
        errors.append(
            f"{path}:{line_number(text, scalar_copy.start())}: "
            "CUDA GramSchmidt still copies one scalar per row to host for norm; "
            "norm should run on GPU."
        )
    return errors


def check_bench_polybench() -> list[str]:
    path = "examples/bench-polybench/src/main.rs"
    text = read(path)
    start, end = block_after_marker(text, "// --- gramschm (NI=NJ=2048) ---")
    body = text[start:end]
    errors = require_tokens(
        text,
        path,
        (
            "pub fn bench_gramschm_kernel1",
            "pub fn bench_gramschm_kernel2(\n    a: &[f32],\n    r: &[f32],",
        ),
    )
    errors.extend(
        require_tokens(
            body,
            path,
            (
                "bench_gramschm_kernel1::launch",
                "&mut r_kk",
                "bench_gramschm_kernel2::launch",
                "&d_r",
            ),
        )
    )
    copy_to_host = re.search(r"\.\s*copy_to_host\s*\(", body)
    if copy_to_host is not None:
        errors.append(
            f"{path}:{line_number(text, start + copy_to_host.start())}: "
            "SeGuRu GramSchmidt benchmark still copies through host inside the "
            "benchmark block; norm should run on GPU."
        )
    return errors


def check_standalone() -> list[str]:
    path = "examples/polybench/gramschm/src/lib.rs"
    text = read(path)
    errors = require_tokens(
        text,
        path,
        (
            "pub fn gramschm_kernel1",
            "pub fn gramschm_kernel2(\n    a: &[f32],\n    r: &[f32],",
        ),
    )
    gpu_start, gpu_end = block_after_marker(text, "// GPU")
    loop_start, loop_end = block_after_pattern(text, r"for\s+k\s+in\s+0\.\.nj\s*\{", gpu_start)
    if loop_end > gpu_end:
        errors.append(f"{path}: GramSchmidt k loop was not contained in the GPU block")
        loop_end = gpu_end
    loop_body = text[loop_start:loop_end]
    errors.extend(
        require_tokens(
            loop_body,
            path,
            (
                "gramschm_kernel1::launch",
                "&mut r_kk",
                "gramschm_kernel2::launch",
                "&d_r",
            ),
        )
    )
    host_copy = re.search(r"\.\s*(copy_to_host|copy_from_host)\s*\(", loop_body)
    if host_copy is not None:
        errors.append(
            f"{path}:{line_number(text, loop_start + host_copy.start())}: "
            "standalone GramSchmidt still copies through host inside the k loop; "
            "norm should run on GPU."
        )
    return errors


def main() -> int:
    errors = check_cuda() + check_bench_polybench() + check_standalone()
    if errors:
        print("\n".join(errors))
        return 1
    print("PolyBench GramSchmidt computes norms on GPU without timed host transfer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
