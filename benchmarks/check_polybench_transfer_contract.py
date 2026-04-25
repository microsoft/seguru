#!/usr/bin/env python3.11
"""RED source check for known-bad PolyBench transfer contracts.

This check is intentionally source-based and fast: it does not run GPU code.
It fails while LU/GramSchmidt benchmark host-device transfer contracts are not
comparable between CUDA and SeGuRu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Block:
    header_start: int
    body_start: int
    body_end: int
    body: str


@dataclass(frozen=True)
class Evidence:
    path: Path
    line: int
    detail: str


@dataclass(frozen=True)
class Finding:
    check: str
    path: Path
    line: int
    message: str


def read_source(relative_path: str) -> tuple[Path, str]:
    path = REPO_ROOT / relative_path
    return path, path.read_text(encoding="utf-8")


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def find_matching_brace(text: str, open_index: int) -> int:
    if text[open_index] != "{":
        raise ValueError("open_index must point at an opening brace")

    depth = 0
    i = open_index
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None
    escaped = False

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
        elif in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                i += 1
        elif in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
        else:
            if char == "/" and next_char == "/":
                in_line_comment = True
                i += 1
            elif char == "/" and next_char == "*":
                in_block_comment = True
                i += 1
            elif char in {"'", '"'}:
                in_string = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    raise ValueError("no matching closing brace found")


def find_block(text: str, header_pattern: str, start: int = 0) -> Block | None:
    match = re.search(header_pattern, text[start:], re.MULTILINE)
    if match is None:
        return None

    header_start = start + match.start()
    open_index = text.find("{", header_start)
    if open_index == -1:
        return None

    close_index = find_matching_brace(text, open_index)
    return Block(
        header_start=header_start,
        body_start=open_index + 1,
        body_end=close_index,
        body=text[open_index + 1 : close_index],
    )


def require_block(path: Path, text: str, header_pattern: str, start: int = 0) -> Block:
    block = find_block(text, header_pattern, start)
    if block is None:
        raise RuntimeError(f"{rel(path)}: unable to find block matching {header_pattern!r}")
    return block


def lu_benchmark_full_matrix_sync() -> Finding | None:
    path, text = read_source("examples/bench-polybench/src/main.rs")
    lu_marker = text.index("// --- lu ---")
    timed_start = text.index("let start = Instant::now();", lu_marker)
    k_loop = require_block(path, text, r"for\s+k\s+in\s+0\.\.n\s*\{", timed_start)

    has_full_d2h = "d_a_write.copy_to_host(&mut h_a_gpu)" in k_loop.body
    has_full_h2d = "d_a_read.copy_from_host(&h_a_gpu)" in k_loop.body
    if not (has_full_d2h and has_full_h2d):
        return None

    copy_index = text.index("d_a_write.copy_to_host(&mut h_a_gpu)", k_loop.body_start)
    return Finding(
        check="lu-seguru-bench-full-matrix-sync",
        path=path,
        line=line_number(text, copy_index),
        message=(
            "LU SeGuRu benchmark copies the full matrix device->host and "
            "host->device inside the timed k loop; CUDA keeps one device "
            "buffer and synchronizes kernels on-device."
        ),
    )


def lu_helper_full_matrix_sync() -> Finding | None:
    path, text = read_source("examples/polybench/lu/src/lib.rs")
    run_lu = text.index("fn run_lu")
    gpu_section = text.index("// GPU", run_lu)
    k_loop = require_block(path, text, r"for\s+k\s+in\s+0\.\.n\s*\{", gpu_section)

    has_full_d2h = "d_a_write\n                    .copy_to_host(&mut h_a_gpu)" in k_loop.body
    has_full_h2d = "d_a_read\n                    .copy_from_host(&h_a_gpu)" in k_loop.body
    if not (has_full_d2h and has_full_h2d):
        return None

    copy_index = text.index(".copy_to_host(&mut h_a_gpu)", k_loop.body_start)
    return Finding(
        check="lu-helper-full-matrix-sync",
        path=path,
        line=line_number(text, copy_index),
        message=(
            "LU helper copies the full matrix through host memory inside "
            "each k iteration, matching the known non-comparable helper "
            "contract."
        ),
    )


def gramschmidt_cuda_scalar_norm_copy() -> Evidence | None:
    path, text = read_source("benchmarks/cuda/bench_gramschm.cu")
    timed_start = text.index("cudaEventRecord(start);")
    k_loop = require_block(
        path,
        text,
        r"for\s*\(\s*int\s+k\s*=\s*0\s*;\s*k\s*<\s*NJ\s*;\s*k\+\+\s*\)\s*\{",
        timed_start,
    )

    scalar_copy = "cudaMemcpy(&col[i]" in k_loop.body
    one_float = "sizeof(float)" in k_loop.body
    d2h = "cudaMemcpyDeviceToHost" in k_loop.body
    column_indexed = "i*NJ+k" in k_loop.body
    if not (scalar_copy and one_float and d2h and column_indexed):
        return None

    copy_index = text.index("cudaMemcpy(&col[i]", k_loop.body_start)
    return Evidence(
        path=path,
        line=line_number(text, copy_index),
        detail="CUDA timed norm path copies one sizeof(float) value per row from d_a[i*NJ+k].",
    )


def gramschmidt_seguru_benchmark_whole_matrix_norm_copy() -> Evidence | None:
    path, text = read_source("examples/bench-polybench/src/main.rs")
    gramschm_marker = text.index("// --- gramschm")
    timed_start = text.index("let start = Instant::now();", gramschm_marker)
    iter_loop = require_block(path, text, r"for\s+_\s+in\s+0\.\.iters\s*\{", timed_start)
    k_loop = require_block(path, text, r"for\s+k\s+in\s+0\.\.nj\s*\{", iter_loop.body_start)

    before_k_loop = text[iter_loop.body_start : k_loop.header_start]
    whole_matrix_buffer = "vec![0.0f32; ni * nj]" in before_k_loop
    full_d2h = "d_a.copy_to_host(&mut h_a_tmp)" in k_loop.body
    if not (whole_matrix_buffer and full_d2h):
        return None

    copy_index = text.index("d_a.copy_to_host(&mut h_a_tmp)", k_loop.body_start)
    return Evidence(
        path=path,
        line=line_number(text, copy_index),
        detail="SeGuRu timed norm path copies the full ni*nj matrix once per column.",
    )


def gramschmidt_helper_whole_matrix_norm_copy() -> Evidence | None:
    path, text = read_source("examples/polybench/gramschm/src/lib.rs")
    run_gramschm = text.index("fn run_gramschm")
    gpu_section = text.index("// GPU", run_gramschm)
    k_loop = require_block(path, text, r"for\s+k\s+in\s+0\.\.nj\s*\{", gpu_section)

    whole_matrix_buffer = "vec![0.0f32; ni * nj]" in k_loop.body
    full_d2h = "d_a.copy_to_host(&mut h_a_tmp)" in k_loop.body
    if not (whole_matrix_buffer and full_d2h):
        return None

    copy_index = text.index("d_a.copy_to_host(&mut h_a_tmp)", k_loop.body_start)
    return Evidence(
        path=path,
        line=line_number(text, copy_index),
        detail="SeGuRu helper norm path copies the full ni*nj matrix inside each k iteration.",
    )


def gramschmidt_granularity_findings() -> list[Finding]:
    findings: list[Finding] = []
    cuda_scalar = gramschmidt_cuda_scalar_norm_copy()
    if cuda_scalar is None:
        return findings

    seguru_bench = gramschmidt_seguru_benchmark_whole_matrix_norm_copy()
    if seguru_bench is not None:
        findings.append(
            Finding(
                check="gramschmidt-bench-d2h-granularity-mismatch",
                path=seguru_bench.path,
                line=seguru_bench.line,
                message=(
                    f"{seguru_bench.detail} {cuda_scalar.detail} "
                    f"CUDA evidence: {rel(cuda_scalar.path)}:{cuda_scalar.line}."
                ),
            )
        )

    seguru_helper = gramschmidt_helper_whole_matrix_norm_copy()
    if seguru_helper is not None:
        findings.append(
            Finding(
                check="gramschmidt-helper-d2h-granularity-mismatch",
                path=seguru_helper.path,
                line=seguru_helper.line,
                message=(
                    f"{seguru_helper.detail} {cuda_scalar.detail} "
                    f"CUDA evidence: {rel(cuda_scalar.path)}:{cuda_scalar.line}."
                ),
            )
        )

    return findings


def collect_findings() -> list[Finding]:
    findings = [
        finding
        for finding in (
            lu_benchmark_full_matrix_sync(),
            lu_helper_full_matrix_sync(),
        )
        if finding is not None
    ]
    findings.extend(gramschmidt_granularity_findings())
    return findings


def main() -> int:
    findings = collect_findings()
    if not findings:
        print("PASS: PolyBench transfer contracts do not match known-bad patterns.")
        return 0

    print(f"FAIL: PolyBench transfer contract check found {len(findings)} issue(s):")
    for finding in findings:
        print(
            f"- [{finding.check}] {rel(finding.path)}:{finding.line}: "
            f"{finding.message}"
        )
    print("Expected RED until LU and GramSchmidt transfer contracts are made comparable.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
