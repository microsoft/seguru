# A trial to build safe CPU-GPU programming in Rust


## Build and Run

### Dependencies

Install homebrew

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

### Run

```bash
cd crates/gpu-test
RUST_LOG=trace RUSTFLAGS="-Zcodegen-backend=`realpath ../target/release/librustc_codegen_gpu.so`" cargo build
```
mlir-opt  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3" /home/ziqiaozhou/rust-gpu/rust-gpu/crates/target/debug/deps/gpu-82572710cbc56735.dikxuq9cyrfdskw44vf8x9rsg.rcgu.bc