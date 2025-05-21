# A trial to build safe CPU-GPU programming in Rust

## Build and Run

### Dependencies to build

Somehow the `melior` lib works well with llvm lib installed via homebrew but not the default one via `apt install`

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)
```

Then install LLVM@20 via 
```
brew install llvm@20
```

### Build codegen backend

```bash
export PATH=`brew --prefix llvm@20`\bin:$PATH
export LD_LIBRARY_PATH=`brew --prefix llvm@20`\bin:$LD_LIBRARY_PATH
cd crates/rustc_codegen_gpu
cargo build --release
```

### Prepare to Run
To run the codegen, you need a custom build of llvm project.

```
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
   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
   -DMLIR_ENABLE_C_BINDINGS=ON \
   -DLINK_POLLY_INTO_TOOLS=ON
ninja
```

### Run

```
export PATH=~/llvm-project/build-mlir-gpu/bin:$PATH
export LD_LIBRARY_PATH=~/llvm-project/build-mlir-gpu/lib:$LD_LIBRARY_PATH
```

```bash
cd crates/gpu-test-basic
RUST_LOG=trace RUSTFLAGS="-Zcodegen-backend=`realpath ../target/release/librustc_codegen_gpu.so`" cargo build

```

You will find target/debug/deps/gpu-xxx.o and it includes a ptx binary in `gpu_bin_cst` symbol. 
TODO: check whether the generated PTX binary is usable.