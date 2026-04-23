use gpu_host::cuda_ctx;
use kernelbench::loss::*;

#[test]
fn test_mse_loss() {
    let n = 256u32;
    let predictions: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
    let targets: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut output = vec![0.0f32; 1];

    cuda_ctx(0, |ctx, m| {
        let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
        let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 256u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(1, 1, 1, block, 1, 1, smem);
        mse_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    let expected: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum::<f32>()
        / n as f32;
    assert!(
        (output[0] - expected).abs() < 1e-3,
        "MSE: got {} expected {}",
        output[0],
        expected
    );
}

#[test]
fn test_huber_loss() {
    let n = 256u32;
    let delta = 1.0f32;
    let predictions: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
    let targets: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.3 - 0.5).collect();
    let mut output = vec![0.0f32; 1];

    cuda_ctx(0, |ctx, m| {
        let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
        let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 256u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(1, 1, 1, block, 1, 1, smem);
        huber_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n, delta)
            .unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    let expected: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            let abs_diff = diff.abs();
            if abs_diff <= delta {
                0.5 * diff * diff
            } else {
                delta * (abs_diff - 0.5 * delta)
            }
        })
        .sum::<f32>()
        / n as f32;
    assert!(
        (output[0] - expected).abs() < 1e-3,
        "Huber: got {} expected {}",
        output[0],
        expected
    );
}

#[test]
fn test_kl_div_loss() {
    let n = 256u32;
    // targets are probabilities (non-negative), predictions are log-probabilities
    let targets: Vec<f32> = (0..n)
        .map(|i| if i % 5 == 0 { 0.0 } else { (i % 7) as f32 * 0.05 + 0.01 })
        .collect();
    let predictions: Vec<f32> = (0..n).map(|i| -((i % 9) as f32 * 0.2 + 0.1)).collect();
    let mut output = vec![0.0f32; 1];

    cuda_ctx(0, |ctx, m| {
        let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
        let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 256u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(1, 1, 1, block, 1, 1, smem);
        kl_div_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    let expected: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| if t > 0.0 { t * (t.ln() - p) } else { 0.0 })
        .sum::<f32>()
        / n as f32;
    assert!(
        (output[0] - expected).abs() < 1e-2,
        "KL Div: got {} expected {}",
        output[0],
        expected
    );
}

#[test]
fn test_hinge_loss() {
    let n = 256u32;
    // predictions are scores, targets are +1 or -1
    let predictions: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.3 - 0.9).collect();
    let targets: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut output = vec![0.0f32; 1];

    cuda_ctx(0, |ctx, m| {
        let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
        let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 256u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(1, 1, 1, block, 1, 1, smem);
        hinge_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    let expected: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (1.0 - p * t).max(0.0))
        .sum::<f32>()
        / n as f32;
    assert!(
        (output[0] - expected).abs() < 1e-3,
        "Hinge: got {} expected {}",
        output[0],
        expected
    );
}
