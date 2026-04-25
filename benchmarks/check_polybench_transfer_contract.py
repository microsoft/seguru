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


def find_block_bounded(
    text: str, header_pattern: str, start: int = 0, end: int | None = None
) -> Block | None:
    if end is None:
        end = len(text)
    if start < 0 or end < start or end > len(text):
        raise ValueError("invalid search bounds")

    search_start = start
    while search_start < end:
        match = re.search(header_pattern, text[search_start:end], re.MULTILINE)
        if match is None:
            return None

        header_start = search_start + match.start()
        open_index = text.find("{", header_start, end)
        if open_index == -1:
            return None

        try:
            close_index = find_matching_brace(text, open_index)
        except ValueError:
            return None
        if close_index < end:
            return Block(
                header_start=header_start,
                body_start=open_index + 1,
                body_end=close_index,
                body=text[open_index + 1 : close_index],
            )

        search_start = header_start + 1

    return None


def find_block(text: str, header_pattern: str, start: int = 0) -> Block | None:
    return find_block_bounded(text, header_pattern, start)


def require_block(path: Path, text: str, header_pattern: str, start: int = 0) -> Block:
    block = find_block(text, header_pattern, start)
    if block is None:
        raise RuntimeError(f"{rel(path)}: unable to find block matching {header_pattern!r}")
    return block


RUST_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

RUST_COPY_TO_HOST = re.compile(
    rf"\b(?P<receiver>{RUST_IDENT})\b\s*"
    rf"\.\s*(?P<method>copy_to_host)\s*"
    rf"\(\s*&\s*mut\s+(?P<buffer>{RUST_IDENT})\b\s*\)"
)
RUST_COPY_FROM_HOST = re.compile(
    rf"\b(?P<receiver>{RUST_IDENT})\b\s*"
    rf"\.\s*(?P<method>copy_from_host)\s*"
    rf"\(\s*&\s*(?P<buffer>{RUST_IDENT})\b\s*\)"
)
CUDA_GRAMSCHMIDT_SCALAR_COPY = re.compile(
    r"cudaMemcpy\s*\(\s*"
    r"&\s*col\s*\[\s*i\s*\]\s*,\s*"
    r"&\s*(?:\(\s*\(\s*float\s*\*\s*\)\s*d_a\s*\)|d_a)\s*"
    r"\[\s*i\s*\*\s*NJ\s*\+\s*k\s*\]\s*,\s*"
    r"sizeof\s*\(\s*float\s*\)\s*,\s*"
    r"cudaMemcpyDeviceToHost\s*\)"
)


def find_host_roundtrip_through_same_buffer(block: Block) -> re.Match[str] | None:
    for d2h in RUST_COPY_TO_HOST.finditer(block.body):
        for h2d in RUST_COPY_FROM_HOST.finditer(block.body, d2h.end()):
            if h2d.group("buffer") == d2h.group("buffer"):
                return d2h
    return None


def find_full_matrix_column_norm_copy(block: Block) -> re.Match[str] | None:
    for copy in RUST_COPY_TO_HOST.finditer(block.body):
        buffer_name = re.escape(copy.group("buffer"))
        column_access = re.compile(
            rf"\b{buffer_name}\s*\[\s*"
            rf"(?:{RUST_IDENT}\s*\*\s*nj\s*\+\s*k|k\s*\+\s*{RUST_IDENT}\s*\*\s*nj)"
            rf"\s*\]"
        )
        if column_access.search(block.body, copy.end()):
            return copy
    return None


def match_line_index(block: Block, match: re.Match[str], group: str = "method") -> int:
    return block.body_start + match.start(group)


def lu_benchmark_full_matrix_sync() -> Finding | None:
    path, text = read_source("examples/bench-polybench/src/main.rs")
    lu_marker = text.index("// --- lu (N=")
    section = require_block(path, text, r"// --- lu \(N=.*?\) ---\s*\{", lu_marker)
    timed_start = text.find("let start = Instant::now();", section.body_start, section.body_end)
    if timed_start == -1:
        return None
    iter_loop = find_block_bounded(
        text, r"for\s+_\s+in\s+0\.\.iters\s*\{", timed_start, section.body_end
    )
    if iter_loop is None:
        return None
    k_loop = find_block_bounded(
        text, r"for\s+k\s+in\s+0\.\.n\s*\{", iter_loop.body_start, iter_loop.body_end
    )
    if k_loop is None:
        return None

    full_sync = find_host_roundtrip_through_same_buffer(k_loop)
    if full_sync is None:
        return None

    copy_index = match_line_index(k_loop, full_sync)
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
    run_lu = require_block(path, text, r"fn\s+run_lu\b[^{]*\{")
    gpu_section = text.find("// GPU", run_lu.body_start, run_lu.body_end)
    if gpu_section == -1:
        return None
    cuda_block = find_block_bounded(
        text, r"cuda_ctx\s*\([^{}]*\|\s*\{", gpu_section, run_lu.body_end
    )
    if cuda_block is None:
        return None
    k_loop = find_block_bounded(
        text, r"for\s+k\s+in\s+0\.\.n\s*\{", cuda_block.body_start, cuda_block.body_end
    )
    if k_loop is None:
        return None

    full_sync = find_host_roundtrip_through_same_buffer(k_loop)
    if full_sync is None:
        return None

    copy_index = match_line_index(k_loop, full_sync)
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
    timed_start = text.find("cudaEventRecord(start);")
    if timed_start == -1:
        return None
    timed_end = text.find("cudaEventRecord(stop);", timed_start)
    if timed_end == -1:
        timed_end = len(text)
    iter_loop = find_block_bounded(
        text,
        r"for\s*\(\s*int\s+it\s*=\s*0\s*;\s*it\s*<\s*ITERS\s*;\s*it\+\+\s*\)\s*\{",
        timed_start,
        timed_end,
    )
    if iter_loop is None:
        return None
    k_loop = find_block_bounded(
        text,
        r"for\s*\(\s*int\s+k\s*=\s*0\s*;\s*k\s*<\s*NJ\s*;\s*k\+\+\s*\)\s*\{",
        iter_loop.body_start,
        iter_loop.body_end,
    )
    if k_loop is None:
        return None

    scalar_copy = CUDA_GRAMSCHMIDT_SCALAR_COPY.search(k_loop.body)
    if scalar_copy is None:
        return None

    copy_index = k_loop.body_start + scalar_copy.start()
    return Evidence(
        path=path,
        line=line_number(text, copy_index),
        detail="CUDA timed norm path copies one sizeof(float) value per row from d_a[i*NJ+k].",
    )


def gramschmidt_seguru_benchmark_whole_matrix_norm_copy() -> Evidence | None:
    path, text = read_source("examples/bench-polybench/src/main.rs")
    gramschm_marker = text.index("// --- gramschm (NI=")
    section = require_block(
        path, text, r"// --- gramschm \(NI=.*?\) ---\s*\{", gramschm_marker
    )
    timed_start = text.find("let start = Instant::now();", section.body_start, section.body_end)
    if timed_start == -1:
        return None
    iter_loop = find_block_bounded(
        text, r"for\s+_\s+in\s+0\.\.iters\s*\{", timed_start, section.body_end
    )
    if iter_loop is None:
        return None
    k_loop = find_block_bounded(
        text, r"for\s+k\s+in\s+0\.\.nj\s*\{", iter_loop.body_start, iter_loop.body_end
    )
    if k_loop is None:
        return None

    full_d2h = find_full_matrix_column_norm_copy(k_loop)
    if full_d2h is None:
        return None

    copy_index = match_line_index(k_loop, full_d2h)
    return Evidence(
        path=path,
        line=line_number(text, copy_index),
        detail="SeGuRu timed norm path copies the full ni*nj matrix once per column.",
    )


def gramschmidt_helper_whole_matrix_norm_copy() -> Evidence | None:
    path, text = read_source("examples/polybench/gramschm/src/lib.rs")
    run_gramschm = require_block(path, text, r"fn\s+run_gramschm\b[^{]*\{")
    gpu_section = text.find("// GPU", run_gramschm.body_start, run_gramschm.body_end)
    if gpu_section == -1:
        return None
    cuda_block = find_block_bounded(
        text, r"cuda_ctx\s*\([^{}]*\|\s*\{", gpu_section, run_gramschm.body_end
    )
    if cuda_block is None:
        return None
    k_loop = find_block_bounded(
        text, r"for\s+k\s+in\s+0\.\.nj\s*\{", cuda_block.body_start, cuda_block.body_end
    )
    if k_loop is None:
        return None

    full_d2h = find_full_matrix_column_norm_copy(k_loop)
    if full_d2h is None:
        return None

    copy_index = match_line_index(k_loop, full_d2h)
    return Evidence(
        path=path,
        line=line_number(text, copy_index),
        detail="SeGuRu helper norm path copies the full ni*nj matrix inside each k iteration.",
    )


def gramschmidt_finding(
    check: str,
    seguru: Evidence,
    cuda_scalar: Evidence | None,
) -> Finding:
    if cuda_scalar is None:
        return Finding(
            check=f"{check}-cuda-evidence-missing",
            path=seguru.path,
            line=seguru.line,
            message=(
                f"{seguru.detail} CUDA scalar-copy evidence was missing or "
                "unrecognized, so this check fails closed instead of passing silently."
            ),
        )

    return Finding(
        check=f"{check}-d2h-granularity-mismatch",
        path=seguru.path,
        line=seguru.line,
        message=(
            f"{seguru.detail} {cuda_scalar.detail} "
            f"CUDA evidence: {rel(cuda_scalar.path)}:{cuda_scalar.line}."
        ),
    )


def gramschmidt_granularity_findings() -> list[Finding]:
    seguru_evidence = [
        ("gramschmidt-bench", gramschmidt_seguru_benchmark_whole_matrix_norm_copy()),
        ("gramschmidt-helper", gramschmidt_helper_whole_matrix_norm_copy()),
    ]
    seguru_evidence = [
        (check, evidence) for check, evidence in seguru_evidence if evidence is not None
    ]
    if not seguru_evidence:
        return []

    cuda_scalar = gramschmidt_cuda_scalar_norm_copy()
    return [
        gramschmidt_finding(check, evidence, cuda_scalar)
        for check, evidence in seguru_evidence
    ]


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
