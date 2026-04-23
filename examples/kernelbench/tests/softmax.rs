use gpu_host::cuda_ctx;
use kernelbench::softmax::*;

#[test]
fn test_softmax_forward() {
    let batch = 4u32;
    let dim = 64u32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.5)
        .collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        softmax_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let row = &input[b * dim as usize..(b + 1) * dim as usize];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        // Verify each row sums to 1
        let row_sum: f32 = output[b * dim as usize..(b + 1) * dim as usize]
            .iter()
            .sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "row {} sum: got {} expected 1.0",
            b,
            row_sum
        );
        for d in 0..dim as usize {
            let expected = (row[d] - max_val).exp() / sum;
            let got = output[b * dim as usize + d];
            assert!(
                (got - expected).abs() < 1e-4,
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
fn test_log_softmax_forward() {
    let batch = 4u32;
    let dim = 64u32;
    let input: Vec<f32> = (0..batch * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.5)
        .collect();
    let mut output = vec![0.0f32; (batch * dim) as usize];

    cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 64u32;
        let smem = block * 4;
        let config = gpu_host::gpu_config!(batch, 1, 1, block, 1, 1, smem);
        log_softmax_forward::launch(config, ctx, m, &d_in, &mut d_out, dim).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });

    for b in 0..batch as usize {
        let row = &input[b * dim as usize..(b + 1) * dim as usize];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
        // Verify exp of log-softmax sums to 1
        let exp_sum: f32 = output[b * dim as usize..(b + 1) * dim as usize]
            .iter()
            .map(|&x| x.exp())
            .sum();
        assert!(
            (exp_sum - 1.0).abs() < 1e-3,
            "row {} exp(log_softmax) sum: got {} expected 1.0",
            b,
            exp_sum
        );
        for d in 0..dim as usize {
            let expected = (row[d] - max_val) - log_sum;
            let got = output[b * dim as usize + d];
            assert!(
                (got - expected).abs() < 1e-4,
                "row {} col {}: got {} expected {}",
                b,
                d,
                got,
                expected
            );
        }
    }
}
