# A trial to build safe CPU-GPU programming in Rust


## Build and Run

### Build codegen backend

```bash
cd crates/rustc_codegen_gpu
cargo build
```

### Run


```bash
cd crates/gpu-test
RUST_LOG=trace RUSTFLAGS="-Zcodegen-backend=`realpath ../target/debug/librustc_codegen_gpu.dylib`" cargo build
```