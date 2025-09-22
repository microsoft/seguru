use std::vec;

use gpu_host::cuda_ctx;
use llm_rs_gpu::gelu_forward_kernel;

use crate::common::random_f32_vec;

mod common;

/// CPU version of GELU forward
pub fn gelu_forward_cpu(inp: &[f32]) -> Vec<f32> {
    let gelu_scaling_factor: f32 = (2.0f32 / core::f32::consts::PI).sqrt();
    let mut out = vec![0.0f32; inp.len()];

    for (o, &x) in out.iter_mut().zip(inp.iter()) {
        let cube = 0.044715f32 * x * x * x;
        *o = 0.5f32 * x * (1.0f32 + (gelu_scaling_factor * (x + cube)).tanh());
    }
    out
}

#[test]
fn test_gelu() {
    const N: usize = 1024;
    let mut h_dinp = random_f32_vec(N);
    let mut h_doutp = [0.0f32; N];
    const BLOCK_SIZE: u32 = 256;
    cuda_ctx(0, |ctx, m| {
        let inp = ctx.new_gmem_with_len(N, &h_dinp).unwrap();
        let outp = ctx.new_gmem_with_len(N, &h_doutp).unwrap();
        let grid_size = N.div_ceil(BLOCK_SIZE as usize);
        let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        gelu_forward_kernel::launch(config, ctx, m, outp, inp, N as i32).expect("launch failed");
        outp.copy_to_host(&mut h_doutp, N, ctx)
            .expect("copy to host failed");
    });
    let expected = gelu_forward_cpu(&h_dinp);
    assert!(
        common::f32_eq(&h_doutp, &expected, 1e-5),
        "h_doutp={:?}\nexpected={:?}",
        &h_doutp[0..32],
        &expected[0..32]
    );
}

pub fn gelu_backward_cpu(inp: &[f32], dout: &[f32]) -> Vec<f32> {
    let gelu_scaling_factor: f32 = (2.0f32 / core::f32::consts::PI).sqrt();
    let mut dinp = vec![0.0f32; inp.len()];
    assert_eq!(dinp.len(), inp.len());
    assert_eq!(dinp.len(), dout.len());
    for ((dx, &x), &dy) in dinp.iter_mut().zip(inp.iter()).zip(dout.iter()) {
        let cube = 0.044715f32 * x * x * x;
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();
        let sech_out = 1.0f32 / (tanh_arg.cosh() * tanh_arg.cosh());
        let local_grad = 0.5 * (1.0 + tanh_out)
            + 0.5 * x * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        *dx = local_grad * dy;
    }
    dinp
}

#[test]
fn test_gelu_backward() {
    const N: usize = 512;
    let h_inp = random_f32_vec(N);
    let h_dout = random_f32_vec(N);
    let mut h_dinp = [0.0f32; N];
    const BLOCK_SIZE: u32 = 128;
    cuda_ctx(0, |ctx, m| {
        let inp = ctx.new_gmem_with_len(N, &h_inp).unwrap();
        let dout = ctx.new_gmem_with_len(N, &h_dout).unwrap();
        let dinp = ctx.new_gmem_with_len(N, &h_dinp).unwrap();
        let grid_size = N.div_ceil(BLOCK_SIZE as usize);
        let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::gelu_backward_kernel::launch(config, ctx, m, dinp, inp, dout, N as i32)
            .expect("launch failed");
        dinp.copy_to_host(&mut h_dinp, N, ctx)
            .expect("copy to host failed");
    });
    let expected = gelu_backward_cpu(&h_inp, &h_dout);
    assert!(
        common::f32_eq(&h_dinp, &expected, 1e-5),
        "h_dinp={:?}\nexpected={:?}",
        &h_dinp[0..32],
        &expected[0..32]
    );
}
