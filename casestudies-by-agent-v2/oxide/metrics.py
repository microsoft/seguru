#!/usr/bin/env python3
"""Lines of device code and `unsafe` counts for the SeGuRu / cuda-oxide pairs.

Only *device* code is counted, on both sides: the host harness, the CPU
reference and the tests differ between the two ports for reasons that have
nothing to do with the safety model, so counting whole files would be
misleading. On the SeGuRu side the device region is every `#[gpu::cuda_kernel]`
and `#[gpu::device]` function; on the cuda-oxide side it is the body of the
`#[cuda_module] mod kernels` block. Blank lines and comment lines are excluded
from the line counts.

Host `unsafe` is reported separately, because cuda-oxide makes
`kernels::load` unsafe unconditionally and that is not a property of any
individual kernel.

Run from anywhere:  ./metrics.py
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASESTUDIES = HERE.parent

# (label, seguru device sources, cuda-oxide source)
PAIRS = [
    (
        "gemm",
        [CASESTUDIES / "polybench/src/gemm.rs"],
        HERE / "polybench-gemm/src/main.rs",
    ),
    (
        "atax",
        [CASESTUDIES / "polybench/src/atax.rs"],
        HERE / "polybench-atax/src/main.rs",
    ),
    (
        "radix_upsweep",
        [
            CASESTUDIES / "gpusorting/src/upsweep.rs",
            CASESTUDIES / "gpusorting/src/utils.rs",
        ],
        HERE / "sort-upsweep/src/main.rs",
    ),
]

DEVICE_ATTR = re.compile(r"^\s*#\[gpu::(cuda_kernel|device)")


def matching_block(lines, start):
    """Return the line range of the brace-delimited block opening at or after
    `start`, as a half-open [first, last) pair of indices into `lines`."""
    depth = 0
    seen = False
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
        if seen and depth == 0:
            return start, i + 1
    return start, len(lines)


def seguru_device_lines(path):
    lines = path.read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        if DEVICE_ATTR.match(lines[i]):
            first, last = matching_block(lines, i)
            out.extend(lines[first:last])
            i = last
        else:
            i += 1
    return out


def oxide_device_lines(path):
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*mod kernels\s*\{", line):
            first, last = matching_block(lines, i)
            return lines[first:last]
    return []


def oxide_host_lines(path):
    lines = path.read_text().splitlines()
    device = set()
    for i, line in enumerate(lines):
        if re.match(r"^\s*mod kernels\s*\{", line):
            first, last = matching_block(lines, i)
            device = set(range(first, last))
            break
    return [l for i, l in enumerate(lines) if i not in device]


def code_lines(lines):
    """Non-blank, non-comment lines."""
    n = 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        n += 1
    return n


def count_unsafe(lines):
    """Occurrences of the `unsafe` keyword, ignoring comments."""
    n = 0
    for line in lines:
        s = line.strip()
        if s.startswith("//"):
            continue
        code = s.split("//")[0]
        n += len(re.findall(r"\bunsafe\b", code))
    return n


def main():
    missing = [p for _, srcs, ox in PAIRS for p in list(srcs) + [ox] if not p.exists()]
    if missing:
        for p in missing:
            print(f"missing: {p}", file=sys.stderr)
        return 1

    print("# Device code size and `unsafe` count")
    print()
    print("Device code only: SeGuRu's `#[gpu::cuda_kernel]` / `#[gpu::device]`")
    print("functions against the body of cuda-oxide's `#[cuda_module] mod kernels`.")
    print("Blank and comment lines are excluded.")
    print()
    print("| Kernel | SeGuRu LoC | SeGuRu `unsafe` | cuda-oxide LoC | cuda-oxide `unsafe` |")
    print("|---|---|---|---|---|")
    for label, seguru_srcs, oxide_src in PAIRS:
        sg = []
        for p in seguru_srcs:
            sg.extend(seguru_device_lines(p))
        ox = oxide_device_lines(oxide_src)
        print(
            f"| {label} | {code_lines(sg)} | {count_unsafe(sg)} "
            f"| {code_lines(ox)} | {count_unsafe(ox)} |"
        )

    print()
    print("## Host-side `unsafe` in the cuda-oxide ports")
    print()
    print("`kernels::load` is an `unsafe fn` in cuda-oxide regardless of what the")
    print("kernel does, so it is counted separately from the kernel bodies.")
    print()
    print("| Benchmark | host `unsafe` |")
    print("|---|---|")
    for label, _, oxide_src in PAIRS:
        print(f"| {label} | {count_unsafe(oxide_host_lines(oxide_src))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
