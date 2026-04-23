use gpu_host::cuda_ctx;
use kernelbench::elementwise::*;

fn grid_block_for(n: u32) -> (u32, u32) {
    let block = 256u32;
    let grid = (n + block - 1) / block;
    (grid, block)
}

#[test]
fn test_relu() {
    let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        relu_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "relu [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_leaky_relu() {
    let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0];
    let alpha = 0.01f32;
    let expected: Vec<f32> = input.iter().map(|&x| if x > 0.0 { x } else { alpha * x }).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        leaky_relu_forward::launch(config, ctx, m, &d_in, &mut d_out, n, alpha).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "leaky_relu [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_sigmoid() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 10.0];
    let expected: Vec<f32> = input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        sigmoid_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "sigmoid [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_tanh() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 10.0];
    let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        tanh_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "tanh [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_swish() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 10.0];
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        swish_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "swish [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_selu() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 2.0];
    let alpha: f32 = 1.6732632;
    let scale: f32 = 1.0507010;
    let expected: Vec<f32> = input.iter().map(|&x| {
        let pos = if x > 0.0 { x } else { 0.0 };
        let neg = if x < 0.0 { alpha * (x.exp() - 1.0) } else { 0.0 };
        scale * (pos + neg)
    }).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        selu_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-3, "selu [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_hard_sigmoid() {
    let input: Vec<f32> = vec![-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0, 0.5];
    let expected: Vec<f32> = input.iter().map(|&x| {
        let val = (x + 3.0) / 6.0;
        if val < 0.0 { 0.0 } else if val > 1.0 { 1.0 } else { val }
    }).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        hard_sigmoid_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "hard_sigmoid [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_softplus() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 5.0];
    let expected: Vec<f32> = input.iter().map(|&x| (1.0f32 + x.exp()).ln()).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        softplus_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "softplus [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_softsign() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 10.0];
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + x.abs())).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        softsign_forward::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "softsign [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_elu() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 2.0];
    let alpha = 1.0f32;
    let expected: Vec<f32> = input.iter().map(|&x| {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    }).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        elu_forward::launch(config, ctx, m, &d_in, &mut d_out, n, alpha).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "elu [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}

#[test]
fn test_hard_tanh() {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, -0.5, 2.0];
    let min_val = -1.0f32;
    let max_val = 1.0f32;
    let expected: Vec<f32> = input.iter().map(|&x| {
        if x < min_val { min_val } else if x > max_val { max_val } else { x }
    }).collect();
    let n = input.len() as u32;
    let mut output = vec![0.0f32; input.len()];
    let (grid, block) = grid_block_for(n);
    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        hard_tanh_forward::launch(config, ctx, m, &d_in, &mut d_out, n, min_val, max_val).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    for i in 0..input.len() {
        assert!((output[i] - expected[i]).abs() < 1e-4, "hard_tanh [{}]: got {} expected {}", i, output[i], expected[i]);
    }
}
