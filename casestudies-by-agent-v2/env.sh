#!/bin/bash
# Source this before building: `source env.sh`
export PATH=/usr/lib/llvm-20/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:${LD_LIBRARY_PATH:-}
export MLIR_SYS_200_PREFIX=/usr/lib/llvm-20
export TABLEGEN_200_PREFIX=/usr/lib/llvm-20
