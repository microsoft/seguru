// KernelBench Level 1 — loss kernels
use gpu::prelude::*;

// KB#94: MSE Loss = mean((pred - target)²)
#[gpu::cuda_kernel(dynamic_shared)]
pub fn mse_loss_forward(predictions: &[f32], targets: &[f32], output: &mut [f32], n: u32) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < n {
        let diff = predictions[i as usize] - targets[i as usize];
        local_sum += diff * diff;
        i += bdim;
    }

    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), (1, 1)] => layout: [i0, t1, t0]),
        );
        out[0] = *smem[0] / n as f32;
    }
}

// KB#96: Huber Loss = mean(huber(pred - target, delta))
// huber(d, delta) = 0.5 * d² if |d| <= delta, else delta * (|d| - 0.5 * delta)
#[gpu::cuda_kernel(dynamic_shared)]
pub fn huber_loss_forward(
    predictions: &[f32],
    targets: &[f32],
    output: &mut [f32],
    n: u32,
    delta: f32,
) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < n {
        let diff = predictions[i as usize] - targets[i as usize];
        let abs_diff = if diff < 0.0 { -diff } else { diff };
        let loss = if abs_diff <= delta {
            0.5 * diff * diff
        } else {
            delta * (abs_diff - 0.5 * delta)
        };
        local_sum += loss;
        i += bdim;
    }

    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), (1, 1)] => layout: [i0, t1, t0]),
        );
        out[0] = *smem[0] / n as f32;
    }
}

// KB#98: KL Divergence Loss = mean(target * (log(target) - prediction))
// Input (predictions) is in log-space; targets are probabilities.
// target=0 contributes 0 to avoid log(0).
#[gpu::cuda_kernel(dynamic_shared)]
pub fn kl_div_loss_forward(predictions: &[f32], targets: &[f32], output: &mut [f32], n: u32) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < n {
        let t = targets[i as usize];
        if t > 0.0 {
            local_sum += t * (GPUDeviceFloatIntrinsics::log(t) - predictions[i as usize]);
        }
        i += bdim;
    }

    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), (1, 1)] => layout: [i0, t1, t0]),
        );
        out[0] = *smem[0] / n as f32;
    }
}

// KB#100: Hinge Loss = mean(max(0, 1 - pred * target))
#[gpu::cuda_kernel(dynamic_shared)]
pub fn hinge_loss_forward(predictions: &[f32], targets: &[f32], output: &mut [f32], n: u32) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < n {
        let v = 1.0 - predictions[i as usize] * targets[i as usize];
        if v > 0.0 {
            local_sum += v;
        }
        i += bdim;
    }

    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), (1, 1)] => layout: [i0, t1, t0]),
        );
        out[0] = *smem[0] / n as f32;
    }
}
