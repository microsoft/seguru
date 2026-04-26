#!/usr/bin/env python3
"""Exercise PolyBench comparison reporting without running GPU benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    "conv2d",
    "conv3d",
    "gemm",
    "twomm",
    "threemm",
    "atax",
    "bicg",
    "mvt",
    "gesummv",
    "syr2k",
    "syrk",
    "corr",
    "covar",
    "doitgen",
    "fdtd2d",
    "gramschm",
    "jacobi1d",
    "jacobi2d",
    "lu",
)


def make_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        benchmarks_dir = tmp / "benchmarks"
        cuda_dir = benchmarks_dir / "cuda"
        examples_dir = tmp / "examples"
        bin_dir = tmp / "bin"
        cuda_home = tmp / "cuda-home"
        for directory in (cuda_dir, examples_dir, bin_dir, cuda_home / "bin"):
            directory.mkdir(parents=True)

        shutil.copy2(ROOT / "benchmarks/run_polybench_comparison.sh", benchmarks_dir)

        for bench in BENCHMARKS:
            (cuda_dir / f"bench_{bench}.cu").write_text("// fake source\n", encoding="utf-8")
            make_executable(
                cuda_dir / f"bench_{bench}",
                f"#!/usr/bin/env bash\necho '{bench} CUDA: 100.000 us/iter (fake)'\n",
            )

        seguru_lines = "\n".join(
            f"{bench} SeGuRu: {99.0 if bench == 'gesummv' else 101.0:.3f} us/iter (fake)"
            for bench in BENCHMARKS
        )
        make_executable(
            bin_dir / "cargo",
            f"""#!/usr/bin/env bash
set -euo pipefail
if [ "${{1:-}}" = "build" ]; then
  echo "fake cargo build"
elif [ "${{1:-}}" = "run" ]; then
  cat <<'EOF'
{seguru_lines}
EOF
else
  echo "unexpected cargo command: $*" >&2
  exit 1
fi
""",
        )
        make_executable(cuda_home / "bin/nvcc", "#!/usr/bin/env bash\nexit 0\n")
        make_executable(bin_dir / "nvidia-smi", "#!/usr/bin/env bash\necho Fake GPU\n")

        env = os.environ.copy()
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        result = subprocess.run(
            ["bash", str(benchmarks_dir / "run_polybench_comparison.sh")],
            cwd=tmp,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        print(result.stdout)
        return result.returncode

    if re.search(r"(?m)\s\.\d{3}x\b", result.stdout):
        print("ratio below 1.0 is missing its leading zero")
        return 1
    if "0.990x" not in result.stdout:
        print(result.stdout)
        print("expected formatted ratio 0.990x was not found")
        return 1

    print("PolyBench report ratios include leading zeroes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
