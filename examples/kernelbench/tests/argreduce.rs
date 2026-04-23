use gpu_host::cuda_ctx;
use kernelbench::argreduce::*;

#[test]
fn test_argmax_reduce() {
    let batch = 4u32;
    let dim = 64u32;
    let mut input = vec![0.0f32; (batch * dim) as usize];
    let expected_indices: Vec<u32> = vec![10, 30, 50, 5];
    for b in 0..batch {
        for d in 0..dim {
            input[(b * dim + d) as usize] = (d % 7) as f32 * 0.1;
        }
        input[(b * dim + expected_indices[b as usize]) as usize] = 100.0;
    }
    let mut output = vec![0u32; batch as usize];
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 8;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        argmax_reduce::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for b in 0..batch as usize {
        assert_eq!(
            output[b], expected_indices[b],
            "argmax row {}: got {} expected {}",
            b, output[b], expected_indices[b]
        );
    }
}

#[test]
fn test_argmin_reduce() {
    let batch = 4u32;
    let dim = 64u32;
    let mut input = vec![10.0f32; (batch * dim) as usize];
    let expected_indices: Vec<u32> = vec![15, 42, 3, 60];
    for b in 0..batch {
        for d in 0..dim {
            input[(b * dim + d) as usize] = (d % 11) as f32 + 5.0;
        }
        input[(b * dim + expected_indices[b as usize]) as usize] = -100.0;
    }
    let mut output = vec![0u32; batch as usize];
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 8;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        argmin_reduce::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for b in 0..batch as usize {
        assert_eq!(
            output[b], expected_indices[b],
            "argmin row {}: got {} expected {}",
            b, output[b], expected_indices[b]
        );
    }
}
