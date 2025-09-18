#![no_std]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use gpu::{chunk_mut, float4, CacheStreamLoadStore, MapLinear};

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

#[gpu_macros::cuda_kernel]
pub fn encoder_forward_kernel3(
    out: &mut [float4],
    inp: &[i32],
    wte: &[float4],
    wpe: &[float4],
    B: i32,
    T: i32,
    C: i32,
) {
    let mut out = chunk_mut(out, MapLinear::new(1));
    let C4 = C / 4;
    let idx = (gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as i32;
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

/// really bad naive kernel with atomicAdd
#[gpu_macros::cuda_kernel]
pub fn encoder_backward_kernel(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: i32,
    T: i32,
    C: i32,
) {
    let idx = (gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as i32;
    let N = B * T * C;
    let dwte = gpu::sync::Atomic::new(dwte);
    let dwpe = gpu::sync::Atomic::new(dwpe);

    if idx < N {
        let bt = idx / C;
        let b = bt / T;
        let t = bt % T;
        let c = idx % C;

        let ix = inp[(b * T + t) as usize];

        let dout_btc: f32 = dout[(b * T * C + t * C + c) as usize];
        dwte.index((ix * C + c) as usize).atomic_addf(dout_btc);
        dwpe.index((t * C + c) as usize).atomic_addf(dout_btc);
    }
}

// TODO: out is column-chunked by wrap. Not supported currently
// #[gpu_macros::cuda_kernel]
// pub fn layernorm_forward_kernel3(
//     out: &mut [f32],
//     mean: &mut [f32],
//     rstd: &mut [f32],
//     inp: &[i32],
//     weight: &[f32],
//     bias: &[f32],
//     N: i32,
//     C: i32,
// ) {
//     unimplemented!();
// }

#[gpu_macros::cuda_kernel]
pub fn permute_kernel(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    inp: &[f32],
    B: i32,
    N: i32,
    NH: i32,
    d: i32,
) {
    let mut q = chunk_mut(q, MapLinear::new(1));
    let mut k = chunk_mut(k, MapLinear::new(1));
    let mut v = chunk_mut(v, MapLinear::new(1));
    let idx = (gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as i32;

    if idx < B * NH * N * d {
        let b = idx / (NH * N * d);
        let mut rest = idx % (NH * N * d);
        let nh_ = rest / (N * d);
        rest = rest % (N * d);
        let n = rest / d;
        let d_ = rest % d;
        let inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;

        // q, k, v are local
        q[0] = inp[(inp_idx) as usize].ldcs();
        k[0] = inp[(inp_idx + NH * d) as usize].ldcs();
        v[0] = inp[(inp_idx + 2 * (NH * d)) as usize].ldcs();
    }
}
