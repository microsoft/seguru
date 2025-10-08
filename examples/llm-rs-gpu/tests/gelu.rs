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
    const N: u32 = 1024;
    let h_dinp = random_f32_vec(N as usize);
    let mut h_doutp = [0.0f32; N as usize];
    const BLOCK_SIZE: u32 = 256;
    cuda_ctx(0, |ctx, m| {
        let inp = ctx.new_tensor_view(h_dinp.as_slice()).unwrap();
        let mut outp = ctx.new_tensor_view(h_doutp.as_slice()).unwrap();
        let grid_size = N.div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(grid_size, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        gelu_forward_kernel::launch(config, ctx, m, &mut outp, &inp, N).expect("launch failed");
        outp.copy_to_host(&mut h_doutp)
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
    let gelu_scaling_factor: f32 = (2.0f32 / core::f32::consts::PI).sqrt(); // sqrtf(2.0f / M_PI)
    let mut dinp = vec![0.0f32; inp.len()];
    assert_eq!(dinp.len(), inp.len());
    for ((dinp_i, &x), &out) in dinp.iter_mut().zip(inp.iter()).zip(dout.iter()) {
        let cube = 0.044715f32 * x * x * x;
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();
        let cosh_out = tanh_arg.cosh();
        let sech_out = 1.0f32 / (cosh_out * cosh_out);
        let local_grad = 0.5 * (1.0 + tanh_out)
            + 0.5 * x * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        *dinp_i = local_grad * out;
    }
    dinp
}

#[test]
fn test_gelu_backward() {
    const N: u32 = 512;
    let mut h_inp = random_f32_vec(N as usize);
    let mut h_dinp = random_f32_vec(N as usize);
    h_dinp[0] = -0.000001009;
    h_inp[0] = -0.176_395_06;
    const BLOCK_SIZE: u32 = 128;
    let expected = gelu_backward_cpu(&h_inp, &h_dinp);
    cuda_ctx(0, |ctx, m| {
        let inp = ctx.new_tensor_view(h_inp.as_slice()).unwrap();
        let mut dinp = ctx.new_tensor_view(h_dinp.as_slice()).unwrap();
        let grid_size = N.div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(grid_size, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::gelu_backward_kernel::launch(config, ctx, m, &mut dinp, &inp, N)
            .expect("launch failed");
        dinp.copy_to_host(&mut h_dinp).expect("copy to host failed");
    });
    assert!(common::f32_eq(&[h_dinp[0]], &[-0.000000364], 1e-7));
    assert!(
        common::f32_eq(&h_dinp, &expected, 1e-5),
        "h_dinp={:?}\nexpected={:?}",
        &h_dinp[0..32],
        &expected[0..32]
    );
}
