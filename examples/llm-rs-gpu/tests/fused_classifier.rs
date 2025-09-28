#![allow(non_snake_case)]

mod common;
use common::{f32_eq, random_f32_vec, random_i32_vec};
use gpu_host::cuda_ctx;

/*
fused_classifier_kernel3

pub fn fused_classifier_kernel3(
    logits: &mut [f32],
    losses: &mut [f32],
    probs: &mut [f32],
    dlosses: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    V: usize, // vocab_size
    P: usize, // padded_vocab_size P >= V
) {
}
*/

#[derive(Debug, Clone, Copy)]
struct SoftmaxParams {
    scale: f32,
    offset: f32,
}

// CPU version of prepare_softmax_blockwide_nofloat4
fn prepare_softmax_row(idx: usize, inp: &[f32], V: usize, P: usize) -> SoftmaxParams {
    let x = &inp[idx * P..(idx + 1) * P];

    let mut thread_maxval = f32::NEG_INFINITY;
    let mut thread_sumval = 0.0;

    for i in (0..V).rev() {
        let v = x[i];
        let old_maxval = thread_maxval;
        thread_maxval = thread_maxval.max(v);
        thread_sumval *= (old_maxval - thread_maxval).exp();
        thread_sumval += (v - thread_maxval).exp();
    }

    let block_maxval = thread_maxval;
    thread_sumval *= (thread_maxval - block_maxval).exp();
    let block_sumval = thread_sumval;

    SoftmaxParams {
        scale: 1.0 / block_sumval,
        offset: block_maxval,
    }
}

#[allow(clippy::too_many_arguments)]
fn fused_classifier_cpu(
    logits: &mut [f32],
    losses: &mut [f32], // size B * T
    probs: &mut [f32],  // size B * T * V
    dlosses: &[f32],
    targets: &[i32], // size B * T
    B: usize,
    T: usize,
    V: usize,
    P: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let idx = b * T + t; // linear index for this (b,t)
            let ix = targets[idx] as usize;

            // softmax
            let sp = prepare_softmax_row(idx, logits, V, P);

            // compute loss
            let logit_idx = idx * P + ix;
            let prob = (logits[logit_idx] - sp.offset).exp() * sp.scale;
            losses[idx] = -prob.ln();

            let dloss = if dlosses.is_empty() {
                1.0 / (B * T) as f32
            } else {
                dlosses[idx]
            };

            // compute gradients and probs
            // compute gradients and probs
            for i in 0..V {
                let logit_pos = idx * P + i;
                let v = logits[logit_pos];
                let prob = (v - sp.offset).exp() * sp.scale;

                if !probs.is_empty() {
                    probs[logit_pos] = prob;
                }

                let indicator = if i == ix { 1.0 } else { 0.0 };
                logits[logit_pos] = (prob - indicator) * dloss;
            }
        }
    }
}

fn test_fused_classifier_kernel3(
    batch: usize,
    t_len: usize,
    vocab_len: usize,
    padded_vocab_len: usize,
    block_size: usize,
) {
    let bt_len = batch * t_len;
    let bt_padded_len: usize = bt_len * padded_vocab_len;
    let mut h_logits = random_f32_vec(bt_padded_len);
    let h_targets = random_i32_vec(bt_len)
        .iter()
        .map(|x| x.abs() % (padded_vocab_len as i32))
        .collect::<Vec<_>>();
    let h_dlosses = random_f32_vec(bt_len);
    let mut h_losses = random_f32_vec(bt_len);
    let mut h_probs = random_f32_vec(bt_len * vocab_len);
    let mut expected_logits = h_logits.clone();
    let mut expected_losses = h_losses.clone();
    let mut expected_probs = h_probs.clone();
    fused_classifier_cpu(
        &mut expected_logits,
        &mut expected_losses,
        &mut expected_probs,
        &h_dlosses,
        &h_targets,
        batch,
        t_len,
        vocab_len,
        padded_vocab_len,
    );
    cuda_ctx(0, |ctx, m| {
        let config = gpu_host::gpu_config!(bt_len as u32, 1, 1, block_size as u32, 1, 1, 0);
        let mut logits = ctx.new_tensor_view::<[f32]>(h_logits.as_slice()).unwrap();
        let mut losses = ctx.new_tensor_view(h_losses.as_slice()).unwrap();
        let mut probs = ctx.new_tensor_view(h_probs.as_slice()).unwrap();

        let targets = ctx.new_tensor_view(h_targets.as_slice()).unwrap();
        let dlosses = ctx.new_tensor_view(h_dlosses.as_slice()).unwrap();
        llm_rs_gpu::fused_classifier_kernel3::launch(
            config,
            ctx,
            m,
            &mut logits,
            &mut losses,
            &mut probs,
            &dlosses,
            &targets,
            batch as _,
            t_len as _,
            vocab_len as _,
            padded_vocab_len as _,
        )
        .expect("Failed to run host arithmetic");
        logits.copy_to_host(&mut h_logits).unwrap();
        losses.copy_to_host(&mut h_losses).unwrap();
        if !h_probs.is_empty() {
            probs.copy_to_host(&mut h_probs).unwrap();
        }
    });

    assert!(
        f32_eq(&h_losses, &expected_losses, 1e-5),
        "losses not match:\n\n{:?}\n\n{:?}",
        &h_losses[0..32.min(bt_len)],
        &expected_losses[0..32.min(bt_len)]
    );
    assert!(
        f32_eq(&h_logits, &expected_logits, 1e-5),
        "logits not match:\n\n{:?}\n\n{:?}",
        &h_logits[0..32.min(bt_padded_len)],
        &expected_logits[0..32.min(bt_padded_len)]
    );

    assert!(
        f32_eq(&h_probs, &expected_probs, 1e-5),
        "probs not match:\n\n{:?}\n\n{:?}",
        &h_probs[0..32.min(bt_padded_len)],
        &expected_probs[0..32.min(bt_padded_len)]
    );
}

#[test]
fn test_fused_classifier_kernel3_small() {
    test_fused_classifier_kernel3(1, 32, 32, 32, 32);
}

#[test]
fn test_fused_classifier_kernel3_large() {
    test_fused_classifier_kernel3(4, 256, 256, 256, 256);
}
