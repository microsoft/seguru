#![allow(non_snake_case)]

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

pub fn softmax_autoregressive_backward_kernel_cpu(
    datt: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    NH: usize,
    scale: f32,
) -> Vec<f32> {
    let mut dpreatt = vec![0f32; NH * B * T * T];
    assert_eq!(dpreatt.len(), NH * B * T * T);
    assert_eq!(datt.len(), NH * B * T * T);
    assert_eq!(att.len(), NH * B * T * T);
    const T_PER_BLOCK: usize = 4;

    let num_time_blocks = 256;

    for block_y in 0..NH * B {
        for block_x in 0..num_time_blocks {
            // CUDA: int idx = blockIdx.y;
            let idx = block_y;

            // reverse traversal
            let t0 = T as isize - 1 - (T_PER_BLOCK as isize) * (block_x as isize);

            // batch offsets
            let att_base = &att[idx * T * T..(idx + 1) * T * T];
            let datt_base = &datt[idx * T * T..(idx + 1) * T * T];
            let dpreatt_base = &mut dpreatt[idx * T * T..(idx + 1) * T * T];

            for to in 0..T_PER_BLOCK {
                let tt = t0 - to as isize;
                if tt < 0 {
                    break;
                }
                let tt = tt as usize;

                let att_row = &att_base[tt * T..(tt + 1) * T];
                let datt_row = &datt_base[tt * T..(tt + 1) * T];
                let dpreatt_row = &mut dpreatt_base[tt * T..(tt + 1) * T];

                // sum_{t2=0..tt} att[t2] * datt[t2]
                let mut local_sum: f64 = 0.0;
                for t2 in 0..=tt {
                    local_sum += att_row[t2] as f64 * datt_row[t2] as f64;
                }

                // dpreatt[t3] = scale * att[t3] * (datt[t3] - local_sum)
                for t3 in 0..=tt {
                    let acc = att_row[t3] as f64 * (datt_row[t3] as f64 - local_sum);
                    dpreatt_row[t3] = (scale as f64 * acc) as f32;
                }
            }
        }
    }
    dpreatt
}

#[test]
fn test_softmax_forward_kernel5_small() {
    test_softmax_forward_kernel5(2, 64);
}

#[test]
fn test_softmax_forward_kernel5_large() {
    test_softmax_forward_kernel5(64, 1024);
}

fn test_softmax_forward_kernel5(N: usize, T: usize) {
    let LEN: usize = N * T * T;
    let inv_temperature = random_f32_vec(1)[0];
    // input: (N, T, T) flattened
    let inp = random_f32_vec(LEN);

    let mut out = vec![0f32; LEN];

    // Reference CPU softmax
    let expected = softmax_cpu(&inp, N, T, inv_temperature);
    const BLOCK_SIZE: u32 = 256;
    let grid_size = (N * T * 32).div_ceil(BLOCK_SIZE as usize) as u32;

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

fn test_softmax_autoregressive_backward_kernel(B: usize, T: usize, C: usize, NH: usize) {
    let scale = 0.4;

    let len: usize = B * NH * T * T;
    // Example att and datt tensors (B, T, T) flattened
    let att = random_f32_vec(len);
    let datt = random_f32_vec(len);
    let mut dpreatt = vec![0f32; len];
    const BLOCK_SIZE: u32 = 256;
    const T_PER_BLOCK: usize = 4;
    let gdim_x = (T / T_PER_BLOCK) as u32;
    let gdim_y = (B * NH) as u32;
    let expected = softmax_autoregressive_backward_kernel_cpu(&datt, &att, B, T, NH, scale);
    cuda_ctx(0, |ctx, m| {
        let d_att = ctx.new_gmem_with_len(len, &att).expect("alloc failed");
        let d_datt = ctx.new_gmem_with_len(len, &datt).expect("alloc failed");
        let d_dpreatt = ctx.new_gmem_with_len(len, &dpreatt).expect("alloc failed");
        let config = gpu_host::gpu_config!(gdim_x, gdim_y, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::softmax_autoregressive_backward_kernel::launch(
            config, ctx, m, d_dpreatt, d_datt, d_att, B, T, C, scale,
        )
        .expect("Failed to run softmax_autoregressive_backward_kernel");
        d_dpreatt
            .copy_to_host(&mut dpreatt, len, ctx)
            .expect("copy to host failed");
    });
    assert!(
        f32_eq(&dpreatt, &expected, 1e-4),
        "dpreatt not match:\n\n{:?}\n\n{:?}",
        &dpreatt[500..520],
        &expected[500..520],
    );
}

#[test]
fn test_softmax_autoregressive_backward_kernel_small_fails() {
    test_softmax_autoregressive_backward_kernel(1, 32, 1, 1);
}

#[test]
fn test_softmax_autoregressive_backward_kernel_large() {
    test_softmax_autoregressive_backward_kernel(2, 1024, 2, 2);
}
