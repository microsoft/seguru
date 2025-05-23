# A trial to build safe CPU-GPU programming in Rust

## Build and Run

### Dependencies to build

Somehow the `melior` lib works well with llvm lib installed via homebrew but not the default one via `apt install`. This might be due to configuration differences. Install Homebrew:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)
```

Then install LLVM@20 via 
```
brew install llvm@20
```

### Build codegen backend

You SHOULD use the Homebrew's LLVM to compile the codegen backend.

```bash
export PATH=`brew --prefix llvm@20`\bin:$PATH
export LD_LIBRARY_PATH=`brew --prefix llvm@20`\bin:$LD_LIBRARY_PATH
cd crates/rustc_codegen_gpu
cargo build --release
```

### Prepare to Run

To run the codegen, you need a custom build of LLVM project. Only 20.1.5 has been tested. CUDA directory must be specified otherwise won't build. Also note that you may want to add the following two lines to `llvm-project/mlir/tools/mlir-shlib/CMakeLists.txt` or it may report compilation error for not linking to the NVIDIA libraries.

```
  target_link_libraries(MLIR PRIVATE MLIR_NVFATBIN_LIB)
  target_link_libraries(MLIR PRIVATE MLIR_NVPTXCOMPILER_LIB)
```

Clone, configure and build:

```
git clone -b llvm-20.1.5 --depth=1 https://github.com/llvm/llvm-project
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="clang;polly;mlir" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DMLIR_ENABLE_CUDA_RUNNER=ON \
   -DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
   -DMLIR_ENABLE_NVPTXCOMPILER=ON \
   -DNVPTX_COMPILER_INCLUDE_DIR=/usr/local/cuda/targets/x86_64-linux/include/ \
   -DNVPTX_COMPILER_LIB_DIR=/usr/local/cuda/targets/x86_64-linux/lib \
   -DCUDACXX=/usr/local/cuda/bin/nvcc \
   -DCUDA_PATH=/usr/local/cuda \
   -DCUDAToolkit_ROOT=/usr/local/cuda \
   -DCUDAToolkit_LIBRARY_ROOT=/usr/local/cuda \
   -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda/lib64 \
   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
   -DMLIR_ENABLE_C_BINDINGS=ON \
   -DLINK_POLLY_INTO_TOOLS=ON \
   -DCMAKE_INSTALL_PREFIX=../../llvm-install \
   -DLLVM_LINK_LLVM_DYLIB=ON
ninja
ninja install
```

### Run

Now use your LLVM as the compiler driver.

```
export PATH=`realpath ../../llvm-install/bin`:$PATH
export LD_LIBRARY_PATH=`realpath ../../llvm-install/lib`:$LD_LIBRARY_PATH
```

Compile your GPU code.

```bash
cd crates/gpu-test-basic
RUST_LOG=trace RUSTFLAGS="-Zcodegen-backend=`realpath ../target/release/librustc_codegen_gpu.so`" cargo build
```

You will find target/debug/deps/gpu-xxx.o and it includes a cubin binary in `gpu_bin_cst` symbol.

The code could be launched by mlir-examples/wrapper.cu.
