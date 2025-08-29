# Safe GPU Programming via Rust

## Building the toolchain

Currently, the project has been tested on Rust `1.87.0-nightly (3f5502370 2025-03-27)`. It'll likely work with a newer version of Rust.

### Install LLVM 20.1.8+

This project requires MLIR which is not yet stable and is under constant evolving. We need at lest LLVM 20.1.8 which unfortunately Ubuntu's APT has yet to catch up. The quickest way to get it is through Homebrew. Install Homebrew via

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

add Homebrew to your PATH

```
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
```

Then install LLVM@20 via 
```bash
brew install llvm@20
```

### Build the toolchain

Remember to put the Homebrew LLVM in the `PATH`:

```bash
export PATH=`brew --prefix llvm@20`/bin:$PATH
export LD_LIBRARY_PATH=`brew --prefix`/lib:`brew --prefix llvm@20`/bin:$LD_LIBRARY_PATH
```

Build the toolchain:

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