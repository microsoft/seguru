use gpu_host::cuda_ctx;
use llm_rs_gpu::softmax_forward_kernel5;
mod common;
use common::{f32_eq, random_f32_vec};

fn softmax_cpu(inp: &[f32], n_len: usize, t_len: usize, inv_temperature: f32) -> Vec<f32> {
    let mut out = vec![0f32; inp.len()];
    for n in 0..n_len {
        for t in 0..t_len {
            let own_pos = t; // causal
            let row_offset = n * t_len * t_len + t * t_len;
            let mut maxval = f32::NEG_INFINITY;
            for i in 0..=own_pos {
                maxval = maxval.max(inp[row_offset + i]);
            }
            let mut sum = 0.0;
            for i in 0..=own_pos {
                sum += (inp[row_offset + i] * inv_temperature).exp();
            }
            for i in 0..=own_pos {
                out[row_offset + i] = (inp[row_offset + i] * inv_temperature).exp() / sum;
            }
        }
    }
    out
}

#[test]
fn test_softmax_forward_kernel5_small() {
    const N: usize = 2; // batch * head
    const T: usize = 4; // sequence length
    let inv_temperature = 1.0;
    const LEN: usize = N * T * T;
    let inp = random_f32_vec(LEN);

    let mut out = [0f32; LEN];

    // Reference CPU softmax
    let expected = softmax_cpu(&inp, N, T, inv_temperature);
    const BLOCK_SIZE: u32 = 256;
    let grid_size = (N * T).div_ceil(BLOCK_SIZE as usize) as u32;

    cuda_ctx(0, |ctx, m| {
        let d_inp = ctx.new_gmem_with_len(LEN, &inp).expect("alloc failed");
        let d_out = ctx.new_gmem_with_len(LEN, &out).expect("alloc failed");
        let config = gpu_host::gpu_config!(grid_size, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        softmax_forward_kernel5::launch(
            config,
            ctx,
            m,
            d_out,
            inv_temperature,
            d_inp,
            N as i32,
            T as i32,
        )
        .expect("Failed to run softmax_forward_kernel5");
        d_out
            .copy_to_host(&mut out, LEN, ctx)
            .expect("copy to host failed");
    });
    assert!(
        f32_eq(&out, &expected, 1e-5),
        "out not match:\n\n{:?}\n\n{:?}",
        &out[0..32],
        &expected[0..32],
    );
}
