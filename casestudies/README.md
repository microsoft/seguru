## Case studies on LLM service and ZKP in Rust

### LLM-RS

This is a Rust-based LLM by converting train_gpt2_fp32.cu to a Rust version.

1. Follow the original LLM.c tutorial to download starter pack.

```bash
cd llm-rs
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2fp32cu
./train_gpt2fp32cu
```

2. Run the Rust version (under llm-rs/rust)

Remember to add `rustc-gpu` into your Path before running it.

```
cd rust
cargo r --release
```

### MSM algorithm in NOVA

```
cd Nova
cargo criterion --bench --no-default-features --features rs_gpu --bench msm-gpu
```
