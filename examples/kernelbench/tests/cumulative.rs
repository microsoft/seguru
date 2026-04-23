use gpu_host::cuda_ctx;
use kernelbench::cumulative::*;

#[test]
fn test_cumsum() {
    let batch = 4u32;
    let dim = 16u32;
    let input: Vec<f32> = (0..batch * dim).map(|i| (i % 5) as f32 + 1.0).collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(batch, 1, 1, 1, 1, 1, 0);
        cumsum_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let mut acc = 0.0f32;
        for d in 0..dim as usize {
            acc += input[b * dim as usize + d];
            assert!(
                (output[b * dim as usize + d] - acc).abs() < 1e-4,
                "row {} col {}: got {} expected {}",
                b,
                d,
                output[b * dim as usize + d],
                acc
            );
        }
    }
}

#[test]
fn test_cumprod() {
    let batch = 2u32;
    let dim = 8u32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| 0.5 + (i % 3) as f32 * 0.2)
        .collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(batch, 1, 1, 1, 1, 1, 0);
        cumprod_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let mut acc = 1.0f32;
        for d in 0..dim as usize {
            acc *= input[b * dim as usize + d];
            assert!(
                (output[b * dim as usize + d] - acc).abs() < 1e-4,
                "row {} col {}: got {} expected {}",
                b,
                d,
                output[b * dim as usize + d],
                acc
            );
        }
    }
}

#[test]
fn test_cumsum_reverse() {
    let batch = 2u32;
    let dim = 8u32;
    let input: Vec<f32> = (0..batch * dim).map(|i| (i % 5) as f32 + 1.0).collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(batch, 1, 1, 1, 1, 1, 0);
        cumsum_reverse_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let mut acc = 0.0f32;
        for d in (0..dim as usize).rev() {
            acc += input[b * dim as usize + d];
            assert!(
                (output[b * dim as usize + d] - acc).abs() < 1e-4,
                "row {} col {}: got {} expected {}",
                b,
                d,
                output[b * dim as usize + d],
                acc
            );
        }
    }
}

#[test]
fn test_cumsum_exclusive() {
    let batch = 2u32;
    let dim = 8u32;
    let input: Vec<f32> = (0..batch * dim).map(|i| (i % 5) as f32 + 1.0).collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(batch, 1, 1, 1, 1, 1, 0);
        cumsum_exclusive_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let mut acc = 0.0f32;
        for d in 0..dim as usize {
            assert!(
                (output[b * dim as usize + d] - acc).abs() < 1e-4,
                "row {} col {}: got {} expected {}",
                b,
                d,
                output[b * dim as usize + d],
                acc
            );
            acc += input[b * dim as usize + d];
        }
    }
}
