use gpu_host::cuda_ctx;
use kernelbench::matvec::*;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < tol,
            "mismatch at [{}]: got {} expected {}",
            i,
            actual[i],
            expected[i]
        );
    }
}

#[test]
fn test_matvec() {
    let (m, n) = (32usize, 64usize);
    let a: Vec<f32> = (0..m * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let x: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1).collect();
    // CPU reference
    let mut expected = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        expected[i] = sum;
    }
    let mut output = vec![0.0f32; m];
    cuda_ctx(0, |ctx, mod_| {
        let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
        let d_x = ctx.new_tensor_view(x.as_slice()).unwrap();
        let mut d_y = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 256u32;
        let grid = (m as u32 + block - 1) / block;
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        matvec_forward::launch(config, ctx, mod_, &d_a, &d_x, &mut d_y, m as u32, n as u32)
            .unwrap();
        d_y.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-3);
}

#[test]
fn test_scalar_multiply() {
    let n = 1024usize;
    let s = 3.14f32;
    let input: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    let expected: Vec<f32> = input.iter().map(|&x| x * s).collect();
    let mut output = vec![0.0f32; n];
    cuda_ctx(0, |ctx, mod_| {
        let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 256u32;
        let grid = (n as u32 + block - 1) / block;
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        scalar_multiply::launch(config, ctx, mod_, &d_in, &mut d_out, s, n as u32).unwrap();
        d_out.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-4);
}

#[test]
fn test_tensor3d_matmul() {
    let (batch, m, n, k) = (2usize, 16usize, 16usize, 8usize);
    let a: Vec<f32> = (0..batch * m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i % 11) as f32 * 0.1).collect();
    // CPU reference: each batch independently
    let mut expected = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[bi * m * k + i * k + l] * b[bi * k * n + l * n + j];
                }
                expected[bi * m * n + i * n + j] = sum;
            }
        }
    }
    let mut output = vec![0.0f32; batch * m * n];
    cuda_ctx(0, |ctx, mod_| {
        let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
        let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let total = (batch * m * n) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
        tensor3d_matmul::launch(
            config,
            ctx,
            mod_,
            &d_a,
            &d_b,
            &mut d_c,
            m as u32,
            n as u32,
            k as u32,
            batch as u32,
        )
        .unwrap();
        d_c.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-2);
}
