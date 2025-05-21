## Example

1. directly compile hello_mlir.mlir to cubin and call the kernel function from wrapper.cu
```
make hello
```

2. Use gpu-basic generated object file and call its kernel function from wrapper.cu

```
cd ../crates/gpu-test-basic/
RUSTFLAGS="-Zcodegen-backend=`realpath ../target/release/librustc_codegen_gpu.so`" cargo build
cd ../../mlir-examples
make gpu_basic
./gpu_basic
```

TODO: Ideally, we want to generate the wrapper code from Rust code marked as `host` and link them without extra steps.