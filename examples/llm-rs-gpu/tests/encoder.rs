#![allow(non_snake_case)]

use gpu_host::cuda_ctx;
use llm_rs_gpu::encoder_backward_kernel;
mod common;
use crate::common::{random_f32_vec, random_i32_vec};
use common::f32_eq;
use gpu::Float4;
use llm_rs_gpu::encoder_forward_kernel3;

pub fn encoder_forward_kernel3_cpu(
    inp: &[i32],
    wte: &[Float4],
    wpe: &[Float4],
    B: i32,
    T: i32,
    C: i32,
) -> Vec<Float4> {
    let C4 = C / 4;
    let N = B * T * C4;
    let mut out = Vec::with_capacity(N as usize);

    for idx in 0..N {
        let bt = idx / C4;
        let b = bt / T;
        let t = bt % T;
        let c4 = idx % C4;
        let ix = inp[(b * T + t) as usize] as usize;

        let val = wte[ix * C4 as usize + c4 as usize] + wpe[t as usize * C4 as usize + c4 as usize];
        out.push(val);
    }

    out
}

pub fn encoder_backward_kernel_cpu(
    dout: &[f32],
    inp: &[i32],
    B: i32,
    T: i32,
    C: i32,
    vocab_size: i32, // needed to size dwte
) -> (Vec<f32>, Vec<f32>) {
    let mut dwte = vec![0.0f32; (vocab_size * C) as usize];
    let mut dwpe = vec![0.0f32; (T * C) as usize];

    let N = B * T * C;

    for idx in 0..N {
        let bt = idx / C;
        let b = bt / T;
        let t = bt % T;
        let c = idx % C;
        let ix = inp[(b * T + t) as usize] as usize;

        let dout_btc = dout[(b * T * C + t * C + c) as usize];
        dwte[ix * C as usize + c as usize] += dout_btc;
        dwpe[t as usize * C as usize + c as usize] += dout_btc;
    }

    (dwte, dwpe)
}

#[test]
fn test_encoder_forward() {
    let batch_size: usize = 2;
    let seq_len: usize = 1024;
    let channel: usize = 16; // must be multiple of 4
    let n = batch_size * seq_len * channel;
    const P: i32 = 64;
    let inp = random_i32_vec(n)
        .iter()
        .map(|x| x.abs() % P)
        .collect::<Vec<_>>();
    let wte_f32 = random_f32_vec(channel * P as usize * 4);
    let wpe_f32 = random_f32_vec(seq_len * channel * 4);
    let mut out = vec![Float4::default(); n / 4];
    let wte = wte_f32
        .chunks(4)
        .map(|x| Float4::new([x[0], x[1], x[2], x[3]]))
        .collect::<Vec<_>>();
    let wpe = wpe_f32
        .chunks(4)
        .map(|x| Float4::new([x[0], x[1], x[2], x[3]]))
        .collect::<Vec<_>>();
    let out_cpu = encoder_forward_kernel3_cpu(
        &inp,
        &wte,
        &wpe,
        batch_size as _,
        seq_len as _,
        channel as _,
    );
    cuda_ctx(0, |ctx, m| {
        let d_inp = ctx.new_tensor_view(inp.as_slice()).expect("alloc failed");
        let mut d_out = ctx.new_tensor_view(out.as_slice()).expect("alloc failed");
        let d_wte = ctx.new_tensor_view(wte.as_slice()).expect("alloc failed");
        let d_wpe = ctx.new_tensor_view(wpe.as_slice()).expect("alloc failed");
        const BSIZE: usize = 512;
        let grid_size = (n / 4).div_ceil(BSIZE);
        let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
        encoder_forward_kernel3::launch(
            config,
            ctx,
            m,
            &mut d_out,
            &d_inp,
            &d_wte,
            &d_wpe,
            batch_size as _,
            seq_len as _,
            channel as _,
        )
        .expect("Failed to run encoder_forward_kernel3");
        d_out.copy_to_host(&mut out).expect("copy to host failed");
    });

    out.iter()
        .zip(out_cpu.iter())
        .for_each(|(a, b)| assert!(f32_eq(&a.data, &b.data, 1e-6), "out {:?} vs {:?}", a, b));
}

#[test]
fn test_encoder_back() {
    let batch_size: usize = 2;
    let seq_len: usize = 256;
    let channel: usize = 16; // must be multiple of 4
    let n = batch_size * seq_len * channel;
    const P: i32 = 64;
    let inp = random_i32_vec(n)
        .iter()
        .map(|x| x.abs() % P)
        .collect::<Vec<_>>();
    let out = random_f32_vec(n);
    let mut wte = vec![0.0f32; channel * P as usize];
    let mut wpe = vec![0.0f32; seq_len * channel];
    let (expected_wte, expected_wpe) = encoder_backward_kernel_cpu(
        &out,
        &inp,
        batch_size as _,
        seq_len as _,
        channel as _,
        P as _,
    );
    cuda_ctx(0, |ctx, m| {
        let d_inp = ctx.new_tensor_view(inp.as_slice()).expect("alloc failed");
        let d_out = ctx.new_tensor_view(out.as_slice()).expect("alloc failed");
        let mut d_wte = ctx.new_tensor_view(wte.as_slice()).expect("alloc failed");
        let mut d_wpe = ctx.new_tensor_view(wpe.as_slice()).expect("alloc failed");

        const BSIZE2: usize = 256;
        let grid_size2 = (n).div_ceil(BSIZE2);
        let config2 = gpu_host::gpu_config!(grid_size2 as u32, 1, 1, @const BSIZE2 as u32, 1, 1, 0);
        encoder_backward_kernel::launch(
            config2,
            ctx,
            m,
            &mut d_wte,
            &mut d_wpe,
            &d_out,
            &d_inp,
            batch_size as _,
            seq_len as _,
            channel as _,
        )
        .expect("Failed to run encoder_backward_kernel");
        d_wte.copy_to_host(&mut wte).expect("copy to host failed");
        d_wpe.copy_to_host(&mut wpe).expect("copy to host failed");
    });

    assert!(
        f32_eq(&wte, &expected_wte, 1e-5),
        "dwte mismatch gpu vs cpu"
    );
    assert!(
        f32_eq(&wpe, &expected_wpe, 1e-5),
        "dwpe mismatch gpu vs cpu"
    );
}
