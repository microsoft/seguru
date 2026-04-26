#!/usr/bin/env python3
"""Keep the CUDA-to-SeGuRu skill doc focused on reusable rules.

The skill doc should be a compact reference for future porting agents. Empirical
history, design decisions, phase summaries, and benchmark progress belong in the
companion progress doc.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DOC = ROOT / "docs/cuda-to-seguru-porting-skill.md"
PROGRESS_DOC = ROOT / "docs/cuda-to-seguru-porting-progress.md"


def main() -> int:
    errors: list[str] = []
    skill = SKILL_DOC.read_text()

    forbidden = [
        "## Case Study:",
        "\n### Phase ",
        "Phase B",
        "Phase C",
        "Phase T",
        "phase B",
        "phase C",
        "skill-doc intervention",
        "LLM",
        "LLM-driven",
        "one-shot",
        "sub-agent",
    ]
    for token in forbidden:
        if token in skill:
            errors.append(f"skill doc still contains history/progress token: {token!r}")

    required_skill_refs = [
        "docs/cuda-to-seguru-porting-progress.md",
        "Raw custom CUDA parity is the primary target",
    ]
    for token in required_skill_refs:
        if token not in skill:
            errors.append(f"skill doc missing required reference/guidance: {token!r}")

    if not PROGRESS_DOC.exists():
        errors.append("missing separated progress doc: docs/cuda-to-seguru-porting-progress.md")
    else:
        progress = PROGRESS_DOC.read_text()
        for token in [
            "## Branch status",
            "## Reference benchmark snapshot",
            "## Design and implementation progress",
            "## Historical notes moved from the skill doc",
            "codegen-i32-addr-arith",
            "gemm_add_relu",
        ]:
            if token not in progress:
                errors.append(f"progress doc missing required content: {token!r}")

    if errors:
        print("\n".join(errors))
        return 1

    print("CUDA-to-SeGuRu docs are split into skill reference and progress history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
