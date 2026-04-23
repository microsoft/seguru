use gpu_host::cuda_ctx;
use kernelbench::norm::*;

#[test]
fn test_rms_norm() {
    let batch = 4u32;
    let dim = 64u32;
    let eps = 1e-5f32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.3)
        .collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        rms_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim, eps).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let row = &input[b * dim as usize..(b + 1) * dim as usize];
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / dim as f32;
        let rms = (mean_sq + eps).sqrt();
        for d in 0..dim as usize {
            let expected = row[d] / rms;
            let got = output[b * dim as usize + d];
            assert!(
                (got - expected).abs() < 1e-3,
                "row {} col {}: got {} expected {}",
                b,
                d,
                got,
                expected
            );
        }
    }
}

#[test]
fn test_frobenius_norm() {
    let n = 256u32;
    let input: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
    let mut output = vec![0.0f32; 1];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 256u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(1, 1, 1, block, 1, 1, smem);
        frobenius_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    let expected: f32 = input.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (output[0] - expected).abs() < 1e-3,
        "got {} expected {}",
        output[0],
        expected
    );
}

#[test]
fn test_l1_norm() {
    let batch = 4u32;
    let dim = 64u32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.3)
        .collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        l1_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let row = &input[b * dim as usize..(b + 1) * dim as usize];
        let sum_abs: f32 = row.iter().map(|&x| x.abs()).sum();
        for d in 0..dim as usize {
            let expected = row[d] / sum_abs;
            let got = output[b * dim as usize + d];
            assert!(
                (got - expected).abs() < 1e-3,
                "row {} col {}: got {} expected {}",
                b,
                d,
                got,
                expected
            );
        }
    }
}

#[test]
fn test_l2_norm() {
    let batch = 4u32;
    let dim = 64u32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.3)
        .collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        l2_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let row = &input[b * dim as usize..(b + 1) * dim as usize];
        let l2: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        for d in 0..dim as usize {
            let expected = row[d] / l2;
            let got = output[b * dim as usize + d];
            assert!(
                (got - expected).abs() < 1e-3,
                "row {} col {}: got {} expected {}",
                b,
                d,
                got,
                expected
            );
        }
    }
}

#[test]
fn test_layer_norm() {
    let batch = 4u32;
    let dim = 64u32;
    let eps = 1e-5f32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.3)
        .collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        layer_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim, eps).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let row = &input[b * dim as usize..(b + 1) * dim as usize];
        let mean: f32 = row.iter().sum::<f32>() / dim as f32;
        let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for d in 0..dim as usize {
            let expected = (row[d] - mean) * inv_std;
            let got = output[b * dim as usize + d];
            assert!(
                (got - expected).abs() < 1e-3,
                "row {} col {}: got {} expected {}",
                b,
                d,
                got,
                expected
            );
        }
    }
}
