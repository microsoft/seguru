use gpu_host::cuda_ctx;

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

/// Due to CPU and GPU precision issue,
/// we use this function to compare two f32 slices.
fn f32_eq(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > 1e-5 {
            return false;
        }
    }
    true
}

pub fn test_layernorm_forward_kernel3<const N: usize, const C: usize, const LEN: usize>(
    input: [f32; LEN],
    weight: [f32; C],
    bias: [f32; C],
) {
    cuda_ctx(0, |ctx, m| {
        assert!(LEN == C * N);
        let h_input = input;
        let h_weight = weight;
        let h_bias = bias;
        let mut h_output = [0.0f32; LEN];
        let mut h_mean = [0.0f32; N];
        let mut h_rstd = [0.0f32; N];
        let (cpu_out, cpu_mean, cpu_rstd) =
            layernorm_forward_cpu(&h_input, &h_weight, &h_bias, N, C);
        let gsize: u32 = N as u32 * 32 / 512;
        let config = gpu_host::gpu_config!(gsize, 1, 1, @const 512, 1, 1, 0);
        let out = ctx.new_gmem_with_len(h_output.len(), &h_output).unwrap();
        let mean = ctx.new_gmem_with_len(h_mean.len(), &h_mean).unwrap();
        let rstd = ctx.new_gmem_with_len(h_rstd.len(), &h_rstd).unwrap();
        let inp = ctx.new_gmem_with_len(h_input.len(), &h_input).unwrap();
        let weight = ctx.new_gmem_with_len(h_weight.len(), &h_weight).unwrap();
        let bias = ctx.new_gmem_with_len(h_bias.len(), &h_bias).unwrap();
        llm_rs_gpu::layernorm_forward_kernel3::launch(
            config, ctx, m, out, mean, rstd, inp, weight, bias, N as _, C as _,
        )
        .expect("Failed to run host arithmetic");
        out.copy_to_host(&mut h_output, LEN, ctx).unwrap();
        mean.copy_to_host(&mut h_mean, N, ctx).unwrap();
        rstd.copy_to_host(&mut h_rstd, N, ctx).unwrap();
        assert!(
            f32_eq(&cpu_rstd, &h_rstd),
            "rstd not match:\n\n{:?}\n\n{:?}",
            &cpu_rstd[0..32],
            &h_rstd[0..32],
        );
        assert!(
            f32_eq(&cpu_mean, &h_mean),
            "mean not match:\n\n{:?}\n\n{:?}",
            &cpu_mean[0..32],
            &h_mean[0..32],
        );
        assert!(
            f32_eq(&cpu_out, &h_output),
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
    test_layernorm_forward_kernel3::<N, C, LEN>([0.0f32; LEN], [1.0f32; C], [0.0f32; C]);
}

#[test]
fn test_basic() {
    const N: usize = 32;
    const C: usize = 32;
    const LEN: usize = N * C;
    let mut input = [0.0f32; LEN];
    for (i, input_elem) in input.iter_mut().enumerate() {
        *input_elem = i as _;
    }
    test_layernorm_forward_kernel3::<N, C, LEN>(input, [1.0f32; C], [0.0f32; C]);
}

#[test]
fn test_large() {
    const N: usize = 128;
    const C: usize = 64;
    const LEN: usize = N * C;
    let mut input = [0.0f32; LEN];
    for (i, input_elem) in input.iter_mut().enumerate() {
        *input_elem = i as _;
    }
    test_layernorm_forward_kernel3::<N, C, LEN>(input, [1.0f32; C], [0.0f32; C]);
}
