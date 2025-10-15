use gpu_host::cuda_ctx;
mod common;
use common::{f32_eq, random_f32_vec};

fn layernorm_forward_cpu(
    inp: &[f32],    // length = LEN
    weight: &[f32], // length = C
    bias: &[f32],   // length = C
    n: usize,
    c: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(inp.len(), n * c);
    assert_eq!(weight.len(), c);
    assert_eq!(bias.len(), c);

    let mut means = vec![0f32; n];
    let mut rstds = vec![0f32; n];
    let mut out = vec![0f32; n * c];

    for row in 0..n {
        let row_base = row * c;
        // compute mean
        let mut sum = 0f32;
        for i in 0..c {
            sum += inp[row_base + i];
        }
        let mean = sum / (c as f32);
        means[row] = mean;

        // compute variance (mean of squared diffs)
        let mut sumsq = 0f32;
        for i in 0..c {
            let d = inp[row_base + i] - mean;
            sumsq += d * d;
        }
        let variance = sumsq / (c as f32) + 1e-5f32;
        let rstd = (1.0f32 / variance).sqrt();
        rstds[row] = rstd;

        // normalize and apply weight/bias
        for i in 0..c {
            let normalized = (inp[row_base + i] - mean) * rstd;
            out[row_base + i] = normalized * weight[i] + bias[i];
        }
    }
    (out, means, rstds)
}

pub fn test_layernorm_forward_kernel3<const N: usize, const C: usize, const LEN: usize>(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) {
    cuda_ctx(0, |ctx, m| {
        assert!(LEN == C * N);
        let h_input = input;
        let h_weight = weight;
        let h_bias = bias;
        let mut h_output = [0.0f32; LEN];
        let mut h_mean = [0.0f32; N];
        let mut h_rstd = [0.0f32; N];
        let (cpu_out, cpu_mean, cpu_rstd) = layernorm_forward_cpu(h_input, h_weight, h_bias, N, C);
        let gsize: u32 = N as u32 * 32 / 512;
        let config = gpu_host::gpu_config!(gsize, 1, 1, @const 512, 1, 1, 0);
        let mut out = ctx.new_tensor_view(h_output.as_slice()).unwrap();
        let mut mean = ctx.new_tensor_view(h_mean.as_slice()).unwrap();
        let mut rstd = ctx.new_tensor_view(h_rstd.as_slice()).unwrap();
        let inp = ctx.new_tensor_view(h_input).unwrap();
        let weight = ctx.new_tensor_view(h_weight).unwrap();
        let bias = ctx.new_tensor_view(h_bias).unwrap();
        llm_rs_gpu::layernorm_forward_kernel3::launch(
            config, ctx, m, &mut out, &mut mean, &mut rstd, &inp, &weight, &bias, N as _, C as _,
        )
        .expect("Failed to run host arithmetic");
        out.copy_to_host(&mut h_output).unwrap();
        mean.copy_to_host(&mut h_mean).unwrap();
        rstd.copy_to_host(&mut h_rstd).unwrap();
        assert!(
            f32_eq(&cpu_rstd, &h_rstd, 1e-5),
            "rstd not match:\n\n{:?}\n\n{:?}",
            &cpu_rstd[0..32],
            &h_rstd[0..32],
        );
        assert!(
            f32_eq(&cpu_mean, &h_mean, 1e-5),
            "mean not match:\n\n{:?}\n\n{:?}",
            &cpu_mean[0..32],
            &h_mean[0..32],
        );
        assert!(
            f32_eq(&cpu_out, &h_output, 1e-5),
            "out not match:\n\n{:?}\n\n{:?}",
            &cpu_out[0..N * 2],
            &h_output[0..N * 2]
        );
    });
}

#[test]
fn test_zero() {
    const N: usize = 32;
    const C: usize = 32;
    const LEN: usize = N * C;
    test_layernorm_forward_kernel3::<N, C, LEN>(&[0.0f32; LEN], &[1.0f32; C], &[0.0f32; C]);
}

#[test]
fn test_basic() {
    const N: usize = 32;
    const C: usize = 32;
    const LEN: usize = N * C;
    let mut input = random_f32_vec(LEN);
    for (i, input_elem) in input.iter_mut().enumerate() {
        *input_elem = i as _;
    }
    test_layernorm_forward_kernel3::<N, C, LEN>(&input, &[1.0f32; C], &[0.0f32; C]);
}

#[test]
fn test_large() {
    const N: usize = 128;
    const C: usize = 64;
    const LEN: usize = N * C;
    let mut input = random_f32_vec(LEN);
    for (i, input_elem) in input.iter_mut().enumerate() {
        *input_elem = i as _;
    }
    test_layernorm_forward_kernel3::<N, C, LEN>(&input, &[1.0f32; C], &[0.0f32; C]);
}

#[allow(clippy::too_many_arguments)]
pub fn layernorm_backward_cpu(
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    batch: usize,
    tlen: usize,
    channel: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dinp = vec![0.0f32; batch * tlen * channel];
    let mut dweight = vec![0.0f32; channel];
    let mut dbias = vec![0.0f32; channel];

    let n = batch * tlen;

    for idx in 0..n {
        let b = idx / tlen;
        let t = idx % tlen;

        let offset = b * tlen * channel + t * channel;
        let dout_bt = &dout[offset..offset + channel];
        let inp_bt = &inp[offset..offset + channel];
        let dinp_bt = &mut dinp[offset..offset + channel];

        let mean_bt = mean[b * tlen + t];
        let rstd_bt = rstd[b * tlen + t];

        // step 1: compute dnorm_mean and dnorm_norm_mean
        let mut dnorm_mean = 0.0;
        let mut dnorm_norm_mean = 0.0;
        for i in 0..channel {
            let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            let dnorm_i = weight[i] * dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean /= channel as f32;
        dnorm_norm_mean /= channel as f32;

        // step 2: accumulate gradients
        for i in 0..channel {
            let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            let dnorm_i = weight[i] * dout_bt[i];

            // bias and weight gradients
            dweight[i] += norm_bti * dout_bt[i];
            dbias[i] += dout_bt[i];
            // input gradient
            let mut dval = dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean;
            dval *= rstd_bt;
            dinp_bt[i] += dval;
        }
    }

    (dinp, dweight, dbias)
}

/*
#[gpu::attr(skip_divergence_check)]
#[gpu::cuda_kernel(dynamic_shared)]
pub fn layernorm_backward_kernel2(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: usize,
    T: usize,
    C: usize,
)
*/
#[test]
fn test_layernorm_backward_kernel2() {
    const B: usize = 2;
    const T: usize = 32;
    const C: usize = 32;
    const BLOCK_SIZE: u32 = 32;
    const SMEM: usize = C * 2 * size_of::<f32>();
    const N: usize = B * T;
    const LEN: usize = N * C;
    let h_dout = random_f32_vec(LEN);
    let h_inp = random_f32_vec(LEN);
    let h_weight = random_f32_vec(N);
    let h_mean = random_f32_vec(N);
    let h_rstd = random_f32_vec(N);
    let mut h_dinp = [0.0f32; LEN];
    let mut h_dweight = [0.0f32; C];
    let mut h_dbias = [0.0f32; C];
    let gsize: u32 = (N as u32 * 32).div_ceil(BLOCK_SIZE);
    cuda_ctx(0, |ctx, m| {
        let config =
            gpu_host::gpu_config!(gsize, 1, 1, @const BLOCK_SIZE, 1, 1, @const (SMEM as u32));
        let mut dinp = ctx.new_tensor_view(h_dinp.as_slice()).unwrap();
        let mut dweight = ctx.new_tensor_view(h_dweight.as_slice()).unwrap();
        let mut dbias = ctx.new_tensor_view(h_dbias.as_slice()).unwrap();
        let dout = ctx.new_tensor_view(h_dout.as_slice()).unwrap();
        let inp = ctx.new_tensor_view(h_inp.as_slice()).unwrap();
        let weight = ctx.new_tensor_view(h_weight.as_slice()).unwrap();
        let mean = ctx.new_tensor_view(h_mean.as_slice()).unwrap();
        let rstd = ctx.new_tensor_view(h_rstd.as_slice()).unwrap();
        llm_rs_gpu::layernorm_backward_kernel2::launch(
            config,
            ctx,
            m,
            &mut dinp,
            &mut dweight,
            &mut dbias,
            &dout,
            &inp,
            &weight,
            &mean,
            &rstd,
            B as _,
            T as _,
            C as _,
        )
        .expect("Failed to run host arithmetic");
        dinp.copy_to_host(&mut h_dinp).unwrap();
        dweight.copy_to_host(&mut h_dweight).unwrap();
        dbias.copy_to_host(&mut h_dbias).unwrap();
    });
    let (expected_dinp, expected_dweight, expected_dbias) =
        layernorm_backward_cpu(&h_dout, &h_inp, &h_weight, &h_mean, &h_rstd, B, T, C);
    assert!(
        f32_eq(&h_dinp, &expected_dinp, 1e-5),
        "dinp not match:\n\n{:?}\n\n{:?}",
        &h_dinp[0..32.min(LEN)],
        &expected_dinp[0..32.min(LEN)]
    );

    assert!(
        f32_eq(&h_dbias, &expected_dbias, 1e-1),
        "dbias not match:\n\n{:?}\n\n{:?}",
        &h_dbias[0..32.min(C)],
        &expected_dbias[0..32.min(C)]
    );
    assert!(
        f32_eq(&h_dweight, &expected_dweight, 1e-5),
        "dweight not match:\n\n{:?}\n\n{:?} \nbut bias matches {:?} \n {:?}",
        &h_dweight[0..32.min(C)],
        &expected_dweight[0..32.min(C)],
        &h_dbias[0..32.min(C)],
        &expected_dbias[0..32.min(C)]
    );
}
