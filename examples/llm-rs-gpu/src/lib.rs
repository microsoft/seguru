#![no_std]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use gpu::float4;

#[gpu_macros::device]
#[inline(always)]
pub fn add_float4(a: &float4, b: &float4) -> float4 {
    float4 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
        w: a.w + b.w,
    }
}

#[gpu_macros::kernel]
#[no_mangle]
pub fn encoder_forward_kernel3(
    out: &gpu::GpuChunkableMut<float4>,
    inp: &[i32],
    wte: &[float4],
    wpe: &[float4],
    B: i32,
    T: i32,
    C: i32,
) {
    let C4 = C / 4;
    let idx = (gpu::block_dim(gpu::DimType::X) * gpu::block_id(gpu::DimType::X)
        + gpu::thread_id(gpu::DimType::X)) as i32;
    let N = B * T * C4;
    if idx < N {
        let bt = idx / C4;
        let b = bt / T;
        let t = bt % T;
        let c4 = idx % C4;
        let ix = inp[(b * T + t) as usize];

        // bt = b * T + t, bt * C4 + c4 = idx
        // b * T * C4 + t * C4 + c4 = idx
        // out is local so we are at 0
        out[0] = add_float4(&wte[(ix * C4 + c4) as usize], &wpe[(t * C4 + c4) as usize]);
    }
}

// really bad naive kernel with atomicAdd
#[gpu_macros::kernel]
#[no_mangle]
pub fn encoder_backward_kernel(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: i32,
    T: i32,
    C: i32,
) {
    let idx = (gpu::block_dim(gpu::DimType::X) * gpu::block_id(gpu::DimType::X)
        + gpu::thread_id(gpu::DimType::X)) as i32;
    let N = B * T * C;

    if idx < N {
        let bt = idx / C;
        let b = bt / T;
        let t = bt % T;
        let c = idx % C;

        let ix = inp[(b * T + t) as usize];

        let dout_btc: f32 = dout[(b * T * C + t * C + c) as usize];

        gpu::atomic_add::<f32>(&mut dwte[(ix * C + c) as usize], dout_btc);
        gpu::atomic_add::<f32>(&mut dwpe[(t * C + c) as usize], dout_btc);
    }
}
