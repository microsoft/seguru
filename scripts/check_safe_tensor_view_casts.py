#!/usr/bin/env python3
"""Reject benchmark/example-local raw TensorView reinterpret casts.

Benchmark and example code should use the checked TensorView cast helpers
(`try_cast_slice` / `try_cast_slice_mut`) instead of spelling raw pointer
reinterprets at call sites.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / "examples", ROOT / "benchmarks"]

RAW_CAST_PATTERNS = [
    re.compile(r"as\s+\*const\s+_\s+as\s+\*const\s+(?:gpu_host::)?TensorView\s*<[^>]*\[[A-Za-z0-9_]+[0-9]\]"),
    re.compile(r"as\s+\*mut\s+_\s+as\s+\*mut\s+(?:gpu_host::)?TensorViewMut\s*<[^>]*\[[A-Za-z0-9_]+[0-9]\]"),
]


def iter_rust_files() -> list[Path]:
    files: list[Path] = []
    for directory in SEARCH_DIRS:
        if directory.exists():
            files.extend(path for path in directory.rglob("*.rs") if path.is_file())
    return sorted(files)


def main() -> int:
    failures: list[str] = []
    for path in iter_rust_files():
        text = path.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in RAW_CAST_PATTERNS):
                failures.append(f"{path.relative_to(ROOT)}:{line_no}: {line.strip()}")

    if failures:
        print("Raw TensorView vector reinterpret casts found; use checked cast helpers instead:")
        print("\n".join(failures))
        return 1

    print("No benchmark/example-local raw TensorView vector casts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
