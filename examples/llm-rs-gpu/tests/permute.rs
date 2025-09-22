use gpu_host::cuda_ctx;
mod common;
use crate::common::random_f32_vec;

#[allow(clippy::too_many_arguments)]
fn test_permute_kernel(
    h_dinp: &[f32],
    h_dq: &[f32],
    h_dk: &[f32],
    h_dv: &[f32],
    b_len: i32,
    n_len: i32,
    nh_len: i32,
    d_len: i32,
) {
    assert_eq!(h_dinp.len() as i32, b_len * nh_len * n_len * d_len * 3);
    assert_eq!(h_dq.len() as i32, b_len * nh_len * n_len * d_len);
    assert_eq!(h_dk.len() as i32, b_len * nh_len * n_len * d_len);
    assert_eq!(h_dv.len() as i32, b_len * nh_len * n_len * d_len);
    cuda_ctx(0, |ctx, m| {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = b_len * nh_len * n_len * d_len;
        let num_blocks = (total_threads as u32).div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        // define GPU mem
        let inp = ctx.new_gmem_with_len(h_dinp.len(), h_dinp).unwrap();
        let dq = ctx.new_gmem_with_len(h_dq.len(), h_dq).unwrap();
        let dk = ctx.new_gmem_with_len(h_dk.len(), h_dk).unwrap();
        let dv = ctx.new_gmem_with_len(h_dv.len(), h_dv).unwrap();
        llm_rs_gpu::permute_kernel::launch(
            config, ctx, m, dq, dk, dv, inp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run permute_kernel");
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::permute_kernel_backward::launch(
            config, ctx, m, inp, dq, dk, dv, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run permute_kernel_backward");
        let mut h_dinp2 = vec![0.0f32; h_dinp.len()];
        inp.copy_to_host(&mut h_dinp2, h_dinp.len(), ctx)
            .expect("copy to host failed");
        assert!(
            common::f32_eq(h_dinp, &h_dinp2, 1e-5),
            "h_dinp not match:\n\n{:?}\n\n{:?}",
            &h_dinp[0..32],
            &h_dinp2[0..32],
        );
    });
}

#[test]
fn test_zero() {
    const B: i32 = 2;
    const N: i32 = 4;
    const NH: i32 = 2;
    const D: i32 = 8;
    const LEN: usize = (B * N * NH * D) as usize;

    let mut h_dinp = [0.0f32; LEN * 3];
    for (i, input) in h_dinp.iter_mut().enumerate() {
        *input = i as f32;
    }
    let h_dq = [0.0f32; LEN];
    let h_dk = [0.0f32; LEN];
    let h_dv = [0.0f32; LEN];
    test_permute_kernel(&h_dinp, &h_dq, &h_dk, &h_dv, B, N, NH, D);
}

#[test]
fn test_basic() {
    const B: i32 = 2;
    const N: i32 = 4;
    const NH: i32 = 2;
    const D: i32 = 8;
    const LEN: usize = (B * N * NH * D) as usize;

    let mut h_dinp = random_f32_vec(LEN * 3);
    for (i, input) in h_dinp.iter_mut().enumerate() {
        *input = i as f32;
    }
    let h_dq = random_f32_vec(LEN);
    let h_dk = random_f32_vec(LEN);
    let h_dv = random_f32_vec(LEN);
    test_permute_kernel(&h_dinp, &h_dq, &h_dk, &h_dv, B, N, NH, D);
}

#[allow(clippy::too_many_arguments)]
fn test_unpermute_kernel(
    h_dinp: &[f32],
    h_doutp: &[f32],
    b_len: i32,
    n_len: i32,
    nh_len: i32,
    d_len: i32,
) {
    assert_eq!(h_dinp.len() as i32, b_len * nh_len * n_len * d_len);
    assert_eq!(h_doutp.len() as i32, b_len * n_len * nh_len * d_len);
    cuda_ctx(0, |ctx, m| {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = b_len * nh_len * n_len * d_len;
        let num_blocks = (total_threads as u32).div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        // define GPU mem
        let inp = ctx.new_gmem_with_len(h_dinp.len(), h_dinp).unwrap();
        let outp = ctx.new_gmem_with_len(h_doutp.len(), h_doutp).unwrap();
        llm_rs_gpu::unpermute_kernel::launch(
            config, ctx, m, inp, outp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run unpermute_kernel");
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::unpermute_kernel_backward::launch(
            config, ctx, m, inp, outp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run unpermute_kernel_backward");
        let mut h_dinp2 = vec![0.0f32; h_dinp.len()];
        inp.copy_to_host(&mut h_dinp2, h_dinp.len(), ctx)
            .expect("copy to host failed");
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

    let mut h_dinp = random_f32_vec(LEN);
    let h_doutp = random_f32_vec(LEN);
    test_unpermute_kernel(&h_dinp, &h_doutp, B, N, NH, D);
}
