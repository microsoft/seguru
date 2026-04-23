use gpu_host::cuda_ctx;
use kernelbench::reduction::*;

#[test]
fn test_sum_reduce() {
    let batch = 4u32;
    let dim = 128u32;
    let input: Vec<f32> = (0..batch * dim).map(|i| (i % 13) as f32 * 0.1).collect();
    let mut output = vec![0.0f32; batch as usize];
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 128u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        sum_reduce::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for b in 0..batch as usize {
        let expected: f32 = input[b * dim as usize..(b + 1) * dim as usize].iter().sum();
        assert!(
            (output[b] - expected).abs() < 1e-2,
            "sum row {}: got {} expected {}",
            b,
            output[b],
            expected
        );
    }
}

#[test]
fn test_mean_reduce() {
    let batch = 4u32;
    let dim = 128u32;
    let input: Vec<f32> = (0..batch * dim).map(|i| (i % 13) as f32 * 0.1).collect();
    let mut output = vec![0.0f32; batch as usize];
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 128u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        mean_reduce::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for b in 0..batch as usize {
        let sum: f32 = input[b * dim as usize..(b + 1) * dim as usize].iter().sum();
        let expected = sum / dim as f32;
        assert!(
            (output[b] - expected).abs() < 1e-2,
            "mean row {}: got {} expected {}",
            b,
            output[b],
            expected
        );
    }
}

#[test]
fn test_max_reduce() {
    let batch = 4u32;
    let dim = 128u32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| (i % 13) as f32 * 0.1 - 0.5)
        .collect();
    let mut output = vec![0.0f32; batch as usize];
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 128u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        max_reduce::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for b in 0..batch as usize {
        let expected = input[b * dim as usize..(b + 1) * dim as usize]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (output[b] - expected).abs() < 1e-4,
            "max row {}: got {} expected {}",
            b,
            output[b],
            expected
        );
    }
}

#[test]
fn test_min_reduce() {
    let batch = 4u32;
    let dim = 128u32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| (i % 13) as f32 * 0.1 - 0.5)
        .collect();
    let mut output = vec![0.0f32; batch as usize];
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 128u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        min_reduce::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for b in 0..batch as usize {
        let expected = input[b * dim as usize..(b + 1) * dim as usize]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        assert!(
            (output[b] - expected).abs() < 1e-4,
            "min row {}: got {} expected {}",
            b,
            output[b],
            expected
        );
    }
}
