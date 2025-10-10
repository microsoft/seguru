# Safe GPU Programming via Rust

This is the source code repository for Safe GPU programming in Rust, a toolchain which aims to provide a easy-to-use and safe GPU language in Rust.

## Design of the tool

![design](./doc/design.png)


## Build the tool (`rustc-gpu`)

1. Add dependencies (cuda, llvm)

```bash
source ./scripts/deps.sh
```

2. Build the tool

```bash
cd crates
cargo build
```

Refer to [install](doc/install.md) for more information about how to build the tool.

## Examples

Add `gpu_macro` and `gpu` crates in Cargo.toml.
```TOML
[dependencies]
gpu_macros = {...}
gpu = {...}
gpu_host = {...}
```

Define GPU and Host functions.

```rust
#[gpu_macros::kernel]
pub fn kernel(input: &[f32; 1]) {
    gpu::println!("Hello world... input = {}", input[0]);
}

#[gpu_macros::host(kernel)]
pub fn host(input: &gpu_host::CudaMemBox<[f32; 1]>) {}

fn main() {
    gpu_host::cuda_ctx(0, |ctx, m| {
        let input = ctx.new_gmem([1.01; 1]).expect("Failed to allocate input");
        let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
        host(config, ctx, m, &input).expect("Failed to run host arithmetic");
    });
}
```

Refer to [examples](examples/) to get more example codes.

```
cd examples
cargo run --bin ...
```

## Tests
