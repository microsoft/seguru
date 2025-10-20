#![allow(non_snake_case)]

use gpu_host::cuda_ctx;
use llm_rs_gpu::softmax_forward_kernel5;
mod common;
use common::{f32_eq, random_f32_vec, random_float4_vec};
use gpu::prelude::*;

fn softmax_cpu(inp: &[f32], n_len: u32, t_len: u32, inv_temperature: f32) -> Vec<f32> {
    let n_len = n_len as usize;
    let t_len = t_len as usize;
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
    B: u32,
    T: u32,
    NH: u32,
    scale: f32,
) -> Vec<f32> {
    let B = B as usize;
    let T = T as usize;
    let NH = NH as usize;

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

fn test_softmax_forward_kernel5(N: u32, T: u32) {
    let LEN: usize = (N * T * T) as usize;
    let inv_temperature = random_f32_vec(1)[0];
    // input: (N, T, T) flattened
    let inp = random_float4_vec(LEN / 4);
    let inp_ref = inp.as_slice();
    let inp_f32 = inp_ref.flatten();

    let mut out = vec![0f32; LEN];

    // Reference CPU softmax
    let expected = softmax_cpu(inp_f32, N, T, inv_temperature);
    const BLOCK_SIZE: u32 = 256;
    let grid_size = (N * T * 32).div_ceil(BLOCK_SIZE);

    cuda_ctx(0, |ctx, m| {
        let d_inp = ctx
            .new_tensor_view::<[gpu::Float4]>(&inp)
            .expect("alloc failed");
        let mut d_out = ctx.new_tensor_view::<[f32]>(&out).expect("alloc failed");
        let config = gpu_host::gpu_config!(grid_size, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        softmax_forward_kernel5::launch(config, ctx, m, &mut d_out, inv_temperature, &d_inp, N, T)
            .expect("Failed to run softmax_forward_kernel5");
        d_out.copy_to_host(&mut out).expect("copy to host failed");
    });
    assert!(
        f32_eq(&out, &expected, 1e-5),
        "out not match:\n\n{:?}\n\n{:?}",
        &out[0..32],
        &expected[0..32],
    );
}

fn test_softmax_autoregressive_backward_kernel(B: u32, T: u32, C: u32, NH: u32) {
    let scale = 0.4;

    let len = (B * NH * T * T) as usize;
    // Example att and datt tensors (B, T, T) flattened
    let att = random_f32_vec(len);
    let datt = random_f32_vec(len);
    let mut dpreatt = vec![0f32; len];
    const BLOCK_SIZE: u32 = 256;
    const T_PER_BLOCK: u32 = 4;
    let gdim_x = T / T_PER_BLOCK;
    let gdim_y = B * NH;
    let expected = softmax_autoregressive_backward_kernel_cpu(&datt, &att, B, T, NH, scale);
    cuda_ctx(0, |ctx, m| {
        let d_att = ctx.new_tensor_view::<[f32]>(&att).expect("alloc failed");
        let d_datt = ctx.new_tensor_view::<[f32]>(&datt).expect("alloc failed");
        let mut d_dpreatt = ctx
            .new_tensor_view::<[f32]>(&dpreatt)
            .expect("alloc failed");
        let config = gpu_host::gpu_config!(gdim_x, gdim_y, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::softmax_autoregressive_backward_kernel::launch(
            config,
            ctx,
            m,
            &mut d_dpreatt,
            &d_datt,
            &d_att,
            B,
            T,
            C,
            scale,
        )
        .expect("Failed to run softmax_autoregressive_backward_kernel");
        d_dpreatt
            .copy_to_host(&mut dpreatt)
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
