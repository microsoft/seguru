use gpu_host::cuda_ctx;
use kernelbench::gelu_variants::*;

fn grid_block_for(n: u32) -> (u32, u32) {
    let block = 256u32;
    let grid = (n + block - 1) / block;
    (grid, block)
}

fn gelu_ref(x: f32) -> f32 {
    let k: f32 = 0.7978845;
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

#[test]
fn test_gelu() {
    let input: Vec<f32> = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| gelu_ref(x)).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        gelu_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "gelu [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_mingpt_new_gelu() {
    let input: Vec<f32> = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| gelu_ref(x)).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        mingpt_new_gelu_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "mingpt_new_gelu [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}
