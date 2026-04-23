// KernelBench Level 1 — elementwise kernels
use gpu::prelude::*;

// KB#19: ReLU — max(0, x)
#[gpu::cuda_kernel]
pub fn relu_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        out[0] = if x > 0.0 { x } else { 0.0 };
    }
}

// KB#20: LeakyReLU — x if x > 0, else alpha * x
#[gpu::cuda_kernel]
pub fn leaky_relu_forward(input: &[f32], output: &mut [f32], n: u32, alpha: f32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        out[0] = if x > 0.0 { x } else { alpha * x };
    }
}

// KB#21: Sigmoid — 1 / (1 + exp(-x))
#[gpu::cuda_kernel]
pub fn sigmoid_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        out[0] = 1.0 / (1.0 + (-x).exp());
    }
}

// KB#22: Tanh — tanh(x)
#[gpu::cuda_kernel]
pub fn tanh_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        out[0] = x.tanh();
    }
}

// KB#25: Swish — x / (1 + exp(-x))
#[gpu::cuda_kernel]
pub fn swish_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        out[0] = x / (1.0 + (-x).exp());
    }
}

// KB#27: SELU — scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
#[gpu::cuda_kernel]
pub fn selu_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        let alpha: f32 = 1.6732632;
        let scale: f32 = 1.0507010;
        let pos = if x > 0.0 { x } else { 0.0 };
        let neg = if x < 0.0 { alpha * (x.exp() - 1.0) } else { 0.0 };
        out[0] = scale * (pos + neg);
    }
}

// KB#28: HardSigmoid — clamp((x + 3) / 6, 0, 1)
#[gpu::cuda_kernel]
pub fn hard_sigmoid_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        let val = (x + 3.0) / 6.0;
        let clamped = if val < 0.0 { 0.0 } else if val > 1.0 { 1.0 } else { val };
        out[0] = clamped;
    }
}

// KB#29: Softplus — log(1 + exp(x))
#[gpu::cuda_kernel]
pub fn softplus_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        let val = 1.0f32 + x.exp();
        out[0] = GPUDeviceFloatIntrinsics::log(val);
    }
}

// KB#30: Softsign — x / (1 + |x|)
#[gpu::cuda_kernel]
pub fn softsign_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        let abs_x = if x < 0.0 { -x } else { x };
        out[0] = x / (1.0 + abs_x);
    }
}

// KB#31: ELU — x if x > 0, else alpha * (exp(x) - 1)
#[gpu::cuda_kernel]
pub fn elu_forward(input: &[f32], output: &mut [f32], n: u32, alpha: f32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        out[0] = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
    }
}

// KB#32: HardTanh — clamp(x, min_val, max_val)
#[gpu::cuda_kernel]
pub fn hard_tanh_forward(input: &[f32], output: &mut [f32], n: u32, min_val: f32, max_val: f32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        let x = input[tid as usize];
        out[0] = if x < min_val { min_val } else if x > max_val { max_val } else { x };
    }
}
