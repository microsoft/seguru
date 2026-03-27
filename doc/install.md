# Safe GPU Programming via Rust

## Building the toolchain

Currently, the project has been tested on Rust `1.87.0-nightly (3f5502370 2025-03-27)`. It'll likely work with a newer version of Rust.

### Install LLVM 20

The project depends on LLVM-20 with MLIR support.

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20
sudo apt-get install libmlir-20-dev mlir-20-tools libpolly-20-dev
export PATH=/usr/lib/llvm-20/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:$LD_LIBRARY_PATH
```

### Build the toolchain

```bash
cd crates
cargo build
```

## Build GPU apps

### Run examples

There are a few examples under the path `examples`. To run, enter the **host** folder of the corresponding example and do

```bash
cargo run --release
```

### Manually test the MLIR generated

While this is not intended for the end user of this project but if you encountered MLIR issue, there's also an `mlir-examples` directory. You can find your MLIR under

```
examples/target/release/deps/xxx.mir
```

Copy that MLIR content to the `mlir-examples/hello_mlir.mlir` and do

```bash
make hello
```

This allows you to play with the MLIR file.