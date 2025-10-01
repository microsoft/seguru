use gpu_host::cuda_ctx;
mod common;
use crate::common::random_f32_vec;

#[allow(clippy::erasing_op)]
pub fn permute_cpu(
    inp: &[f32],
    b_len: usize,
    n_len: usize,
    nh: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Allocate output tensors: shape [B, NH, N, d] flattened
    let mut q = vec![0f32; b_len * nh * n_len * d];
    let mut k = vec![0f32; b_len * nh * n_len * d];
    let mut v = vec![0f32; b_len * nh * n_len * d];

    for idx in 0..(b_len * nh * n_len * d) {
        let b = idx / (nh * n_len * d);
        let rest1 = idx % (nh * n_len * d);
        let nh_ = rest1 / (n_len * d);
        let rest2 = rest1 % (n_len * d);
        let n = rest2 / d;
        let d_ = rest2 % d;

        // Compute the index in inp: inp[b][n][0/1/2][nh_][d_]
        let inp_idx = b * n_len * 3 * nh * d + n * 3 * nh * d + 0 * nh * d + nh_ * d + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + nh * d];
        v[idx] = inp[inp_idx + 2 * nh * d];
    }

    (q, k, v)
}

#[allow(clippy::erasing_op)]
pub fn permute_backward_cpu(
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    b_len: usize,
    n_len: usize,
    nh: usize,
    d: usize,
) -> Vec<f32> {
    // Allocate output tensor: shape [B, N, 3, NH, d] flattened
    let mut dinp = vec![0f32; b_len * n_len * 3 * nh * d];

    for idx in 0..(b_len * nh * n_len * d) {
        let b = idx / (nh * n_len * d);
        let rest1 = idx % (nh * n_len * d);
        let nh_ = rest1 / (n_len * d);
        let rest2 = rest1 % (n_len * d);
        let n = rest2 / d;
        let d_ = rest2 % d;

        let inp_idx = b * n_len * 3 * nh * d + n * 3 * nh * d + 0 * nh * d + nh_ * d + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + nh * d] = dk[idx];
        dinp[inp_idx + 2 * nh * d] = dv[idx];
    }

    dinp
}

#[allow(clippy::needless_range_loop)]
pub fn unpermute_cpu(inp: &[f32], b_len: usize, n_len: usize, nh: usize, d: usize) -> Vec<f32> {
    // Allocate output tensor: shape [B, N, NH, d] flattened
    let mut out = vec![0f32; b_len * n_len * nh * d];

    for idx in 0..(b_len * nh * n_len * d) {
        let b = idx / (nh * n_len * d);
        let rest1 = idx % (nh * n_len * d);
        let nh_ = rest1 / (n_len * d);
        let rest2 = rest1 % (n_len * d);
        let n = rest2 / d;
        let d_ = rest2 % d;

        // Compute the flattened index for output: out[b][n][nh_][d_]
        let other_idx = b * n_len * nh * d + n * nh * d + nh_ * d + d_;
        out[other_idx] = inp[idx]; // __ldcs is just a load on CPU
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn test_permute_kernel(h_dinp: &[f32], b_len: i32, n_len: i32, nh_len: i32, d_len: i32) {
    let len = (b_len * nh_len * n_len * d_len) as usize;
    let mut h_dq: Vec<f32> = random_f32_vec(len);
    let mut h_dk: Vec<f32> = random_f32_vec(len);
    let mut h_dv: Vec<f32> = random_f32_vec(len);
    assert_eq!(h_dinp.len() as i32, b_len * nh_len * n_len * d_len * 3);
    cuda_ctx(0, |ctx, m| {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = b_len * nh_len * n_len * d_len;
        let num_blocks = (total_threads as u32).div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        // define GPU mem
        let mut inp = ctx.new_tensor_view(h_dinp).unwrap();
        let mut dq = ctx.new_tensor_view(h_dq.as_slice()).unwrap();
        let mut dk = ctx.new_tensor_view(h_dk.as_slice()).unwrap();
        let mut dv = ctx.new_tensor_view(h_dv.as_slice()).unwrap();
        llm_rs_gpu::permute_kernel::launch(
            config, ctx, m, &mut dq, &mut dk, &mut dv, &inp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run permute_kernel");
        let (expected_dq, expected_dk, expected_dv) = permute_cpu(
            h_dinp,
            b_len as usize,
            n_len as usize,
            nh_len as usize,
            d_len as usize,
        );
        dq.copy_to_host(&mut h_dq).unwrap();
        dk.copy_to_host(&mut h_dk).unwrap();
        dv.copy_to_host(&mut h_dv).unwrap();
        assert!(
            common::f32_eq(&h_dq, &expected_dq, 1e-8),
            "h_dq not match:\n\n{:?}\n\n{:?}",
            &h_dq[0..32],
            &expected_dq[0..32],
        );
        assert!(
            common::f32_eq(&h_dk, &expected_dk, 1e-8),
            "h_dk not match:\n\n{:?}\n\n{:?}",
            &h_dk[0..32],
            &expected_dk[0..32],
        );
        assert!(
            common::f32_eq(&h_dv, &expected_dv, 1e-8),
            "h_dv not match:\n\n{:?}\n\n{:?}",
            &h_dv[0..32],
            &expected_dv[0..32],
        );

        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::permute_kernel_backward::launch(
            config, ctx, m, &mut inp, &dq, &dk, &dv, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run permute_kernel_backward");
        let mut h_dinp2 = vec![0.0f32; h_dinp.len()];
        inp.copy_to_host(&mut h_dinp2).expect("copy to host failed");
        assert!(
            common::f32_eq(h_dinp, &h_dinp2, 1e-5),
            "h_dinp not match:\n\n{:?}\n\n{:?}",
            &h_dinp[0..32],
            &h_dinp2[0..32],
        );
    });
}

#[test]
fn test_basic() {
    const B: i32 = 2;
    const N: i32 = 4;
    const NH: i32 = 2;
    const D: i32 = 8;
    const LEN: usize = (B * N * NH * D) as usize;

    let mut h_dinp = [0.0f32; LEN * 3];
    for (i, input) in h_dinp.iter_mut().enumerate() {
        *input = i as f32;
    }
    test_permute_kernel(&h_dinp, B, N, NH, D);
}

#[test]
fn test_basic_random() {
    const B: i32 = 2;
    const N: i32 = 4;
    const NH: i32 = 2;
    const D: i32 = 8;
    const LEN: usize = (B * N * NH * D) as usize;

    let h_dinp = random_f32_vec(LEN * 3);
    test_permute_kernel(&h_dinp, B, N, NH, D);
}

#[allow(clippy::too_many_arguments)]
fn test_unpermute_kernel(h_dinp: &[f32], b_len: i32, n_len: i32, nh_len: i32, d_len: i32) {
    let len = (b_len * n_len * nh_len * d_len) as usize;
    assert_eq!(h_dinp.len(), len);
    let mut h_doutp = vec![0.0f32; len];
    cuda_ctx(0, |ctx, m| {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = b_len * nh_len * n_len * d_len;
        let num_blocks = (total_threads as u32).div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        // define GPU mem
        let mut inp = ctx.new_tensor_view(h_dinp).unwrap();
        let mut outp = ctx.new_tensor_view(h_doutp.as_slice()).unwrap();
        llm_rs_gpu::unpermute_kernel::launch(
            config, ctx, m, &inp, &mut outp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run unpermute_kernel");
        let expected = unpermute_cpu(
            h_dinp,
            b_len as usize,
            n_len as usize,
            nh_len as usize,
            d_len as usize,
        );
        outp.copy_to_host(&mut h_doutp).unwrap();
        assert!(
            common::f32_eq(&h_doutp, &expected, 1e-8),
            "h_doutp not match:\n\n{:?}\n\n{:?}",
            &h_doutp[0..32],
            &expected[0..32],
        );

        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::unpermute_kernel_backward::launch(
            config, ctx, m, &mut inp, &outp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run unpermute_kernel_backward");
        let mut h_dinp2 = vec![0.0f32; h_dinp.len()];
        inp.copy_to_host(&mut h_dinp2).expect("copy to host failed");
        assert!(
            common::f32_eq(h_dinp, &h_dinp2, 1e-5),
            "h_dinp not match:\n\n{:?}\n\n{:?}",
            &h_dinp[0..32],
            &h_dinp2[0..32],
        );
    });
}

#[test]
fn test_basic_unpermute() {
    const B: i32 = 4;
    const N: i32 = 8;
    const NH: i32 = 2;
    const D: i32 = 16;
    const LEN: usize = (B * N * NH * D) as usize;

    let h_inp = random_f32_vec(LEN);
    test_unpermute_kernel(&h_inp, B, N, NH, D);
}
