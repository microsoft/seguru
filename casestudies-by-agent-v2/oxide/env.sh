# Environment for the cuda-oxide side of the comparison.
#
# Source this before every `cargo oxide` command in this directory, the same
# way `../env.sh` is sourced before every `cargo` command on the SeGuRu side:
#
#   cd casestudies-by-agent-v2/oxide && source env.sh
#
# CUDA_OXIDE points at a checkout of https://github.com/NVlabs/cuda-oxide that
# has been built at least once (`cargo build -p cargo-oxide` plus one
# `cargo oxide run <example>` to produce the codegen backend). Override it if
# your checkout lives somewhere else:
#
#   CUDA_OXIDE=/path/to/cuda-oxide source env.sh

_oxide_link="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.cuda-oxide"
: "${CUDA_OXIDE:=$_oxide_link}"
export CUDA_OXIDE

if [ ! -d "$CUDA_OXIDE/crates/rustc-codegen-cuda" ]; then
    echo "env.sh: no cuda-oxide checkout at $CUDA_OXIDE" >&2
    echo "env.sh: see oxide/README.md for how to obtain and build one" >&2
    return 1 2>/dev/null || exit 1
fi

# Each crate's path dependencies name `../.cuda-oxide`, which an environment
# variable cannot redirect, so keep the symlink pointing at $CUDA_OXIDE.
if [ "$CUDA_OXIDE" != "$_oxide_link" ]; then
    ln -sfn "$CUDA_OXIDE" "$_oxide_link"
fi
unset _oxide_link

# `cargo oxide` in standalone mode would otherwise clone and rebuild the
# codegen backend from git; point it at the one already built in the checkout.
export CUDA_OXIDE_BACKEND="$CUDA_OXIDE/crates/rustc-codegen-cuda/target/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so"

export PATH="$CUDA_OXIDE/target/debug:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
