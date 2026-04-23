// KernelBench Level 1 — gelu_variants kernels
use gpu::prelude::*;

// KB#26: GELU — 0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[gpu::cuda_kernel]
pub fn gelu_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        let k: f32 = 0.7978845;
        let inner = k * (x + 0.044715 * x * x * x);
        out[0] = 0.5 * x * (1.0 + inner.tanh());
    }
}

// KB#88: MinGPTNewGelu — same formula, kept separate for benchmark identity
#[gpu::cuda_kernel]
pub fn mingpt_new_gelu_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        let k: f32 = 0.7978845;
        let inner = k * (x + 0.044715 * x * x * x);
        out[0] = 0.5 * x * (1.0 + inner.tanh());
    }
}
