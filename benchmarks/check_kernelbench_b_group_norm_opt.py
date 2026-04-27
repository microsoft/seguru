#!/usr/bin/env python3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED = {
    "examples/kernelbench-b/cuda/group_norm.cu": [
        "group_norm_stats_kernel",
        "group_norm_apply_kernel",
        "float4",
        "group_elems4",
        "rstd",
    ],
    "examples/kernelbench-b/src/group_norm.rs": [
        "group_norm_stats_kernel",
        "group_norm_apply_kernel",
        "Float4",
        "group_elems4",
        "chunk_to_scope(grid2block",
        "checked_mul",
        "u32::try_from",
        "group_norm requires non-empty",
    ],
    "examples/kernelbench-b/src/from_cuda/group_norm.rs": [
        "group_norm_stats_kernel",
        "group_norm_apply_kernel",
        "Float4",
        "group_elems4",
        "chunk_to_scope(grid2block",
        "checked_mul",
        "u32::try_from",
        "group_norm requires non-empty",
    ],
}

DOC = "docs/cuda-to-seguru-porting-skill.md"


def main() -> None:
    for rel, tokens in REQUIRED.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        if missing:
            raise AssertionError(f"{rel} missing {missing}")
    doc = (REPO / DOC).read_text(encoding="utf-8")
    start = doc.index("### Pattern: 1 block per row")
    end = doc.index("### Pattern: 1 warp per row", start)
    row_block_example = doc[start:end]
    bad = "let mut out = chunk_mut(y, MapContinuousLinear::new(1));"
    if bad in row_block_example:
        raise AssertionError(f"{DOC} row-reduction example still uses plain chunk_mut")
    required_doc_tokens = [
        "let grid2block = build_chunk_scope(Grid, Block);",
        "let block2thread = build_chunk_scope(Block, Thread);",
        ".chunk_to_scope(grid2block, MapContinuousLinear::new(1))",
        ".chunk_to_scope(block2thread, MapContinuousLinear::new(1))",
    ]
    missing_doc_tokens = [
        token for token in required_doc_tokens if token not in row_block_example
    ]
    if missing_doc_tokens:
        raise AssertionError(f"{DOC} row-reduction example missing {missing_doc_tokens}")
    print("KernelBench-B group_norm optimization guard passed.")


if __name__ == "__main__":
    main()
