#!/bin/bash
# Set up the build environment for SeGuRu.
# Usage: source ./scripts/env.sh

# --- CUDA ---
CUDA_PREFIX="${CUDA_PREFIX:-/usr/local/cuda}"
if [ ! -d "$CUDA_PREFIX" ]; then
    echo "ERROR: CUDA not found at $CUDA_PREFIX"
    echo "Install the CUDA Toolkit or set CUDA_PREFIX to your installation path."
    return 1 2>/dev/null || exit 1
fi

CUDA_VERSION=$("$CUDA_PREFIX/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

if [ "$CUDA_MAJOR" -lt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 8 ]; }; then
    echo "WARNING: CUDA $CUDA_VERSION detected. Version 12.8+ is recommended."
fi

export PATH="$CUDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA $CUDA_VERSION at $CUDA_PREFIX"

# --- LLVM / MLIR ---
LLVM_PREFIX="${LLVM_PREFIX:-/usr/lib/llvm-20}"
if [ ! -d "$LLVM_PREFIX" ]; then
    echo "ERROR: LLVM 20 not found at $LLVM_PREFIX"
    echo "Run ./scripts/deps.sh first, or set LLVM_PREFIX to your LLVM 20 installation."
    return 1 2>/dev/null || exit 1
fi

export PATH="$LLVM_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MLIR_SYS_200_PREFIX="$LLVM_PREFIX"
export TABLEGEN_200_PREFIX="$LLVM_PREFIX"
echo "LLVM $("$LLVM_PREFIX/bin/llvm-config" --version) at $LLVM_PREFIX"

# --- Dev libraries ---
missing_libs=()
if ! ldconfig -p 2>/dev/null | grep -q libz\\.so; then missing_libs+=("zlib1g-dev"); fi
if ! ldconfig -p 2>/dev/null | grep -q libzstd\\.so; then missing_libs+=("libzstd-dev"); fi
if [ ${#missing_libs[@]} -gt 0 ]; then
    echo "WARNING: Missing dev packages: ${missing_libs[*]}"
    echo "Install with: sudo apt-get install ${missing_libs[*]}"
fi

echo "Environment ready. Run 'cargo build' in crates/ to build."
