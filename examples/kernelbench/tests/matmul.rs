use gpu_host::cuda_ctx;
use kernelbench::matmul::*;

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

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
fn test_matmul_square() {
    let n = 64usize;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();
    let expected = cpu_matmul(&a, &b, n, n, n);
    let mut output = vec![0.0f32; n * n];
    cuda_ctx(0, |ctx, m| {
        let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
        let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 16u32;
        let grid = (n as u32 + block - 1) / block;
        let config = gpu_host::gpu_config!(grid, grid, 1, block, block, 1, 0);
        matmul_forward::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32, n as u32, n as u32)
            .unwrap();
        d_c.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-2);
}

#[test]
fn test_matmul_rectangular() {
    let (m, n, k) = (32usize, 48usize, 16usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();
    let expected = cpu_matmul(&a, &b, m, n, k);
    let mut output = vec![0.0f32; m * n];
    cuda_ctx(0, |ctx, mod_| {
        let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
        let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 16u32;
        let grid_x = (n as u32 + block - 1) / block;
        let grid_y = (m as u32 + block - 1) / block;
        let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block, block, 1, 0);
        matmul_forward::launch(
            config,
            ctx,
            mod_,
            &d_a,
            &d_b,
            &mut d_c,
            m as u32,
            n as u32,
            k as u32,
        )
        .unwrap();
        d_c.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-2);
}

#[test]
fn test_matmul_transposed_a() {
    let (m, n, k) = (32usize, 48usize, 16usize);
    // A is stored as K×M (transposed)
    let a_t: Vec<f32> = (0..k * m).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();
    // CPU reference: transpose A back to M×K, then multiply
    let mut a_normal = vec![0.0f32; m * k];
    for r in 0..k {
        for c in 0..m {
            a_normal[c * k + r] = a_t[r * m + c];
        }
    }
    let expected = cpu_matmul(&a_normal, &b, m, n, k);
    let mut output = vec![0.0f32; m * n];
    cuda_ctx(0, |ctx, mod_| {
        let d_a = ctx.new_tensor_view(a_t.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
        let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 16u32;
        let grid_x = (n as u32 + block - 1) / block;
        let grid_y = (m as u32 + block - 1) / block;
        let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block, block, 1, 0);
        matmul_transposed_a::launch(
            config,
            ctx,
            mod_,
            &d_a,
            &d_b,
            &mut d_c,
            m as u32,
            n as u32,
            k as u32,
        )
        .unwrap();
        d_c.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-2);
}

#[test]
fn test_matmul_transposed_b() {
    let (m, n, k) = (32usize, 48usize, 16usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    // B is stored as N×K (transposed)
    let b_t: Vec<f32> = (0..n * k).map(|i| (i % 11) as f32 * 0.1).collect();
    // CPU reference: transpose B back to K×N, then multiply
    let mut b_normal = vec![0.0f32; k * n];
    for r in 0..n {
        for c in 0..k {
            b_normal[c * n + r] = b_t[r * k + c];
        }
    }
    let expected = cpu_matmul(&a, &b_normal, m, n, k);
    let mut output = vec![0.0f32; m * n];
    cuda_ctx(0, |ctx, mod_| {
        let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(b_t.as_slice()).unwrap();
        let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 16u32;
        let grid_x = (n as u32 + block - 1) / block;
        let grid_y = (m as u32 + block - 1) / block;
        let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block, block, 1, 0);
        matmul_transposed_b::launch(
            config,
            ctx,
            mod_,
            &d_a,
            &d_b,
            &mut d_c,
            m as u32,
            n as u32,
            k as u32,
        )
        .unwrap();
        d_c.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-2);
}

#[test]
fn test_matmul_transposed_both() {
    let (m, n, k) = (32usize, 48usize, 16usize);
    // A stored as K×M, B stored as N×K
    let a_t: Vec<f32> = (0..k * m).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_t: Vec<f32> = (0..n * k).map(|i| (i % 11) as f32 * 0.1).collect();
    // Transpose both back for CPU reference
    let mut a_normal = vec![0.0f32; m * k];
    for r in 0..k {
        for c in 0..m {
            a_normal[c * k + r] = a_t[r * m + c];
        }
    }
    let mut b_normal = vec![0.0f32; k * n];
    for r in 0..n {
        for c in 0..k {
            b_normal[c * n + r] = b_t[r * k + c];
        }
    }
    let expected = cpu_matmul(&a_normal, &b_normal, m, n, k);
    let mut output = vec![0.0f32; m * n];
    cuda_ctx(0, |ctx, mod_| {
        let d_a = ctx.new_tensor_view(a_t.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(b_t.as_slice()).unwrap();
        let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
        let block = 16u32;
        let grid_x = (n as u32 + block - 1) / block;
        let grid_y = (m as u32 + block - 1) / block;
        let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block, block, 1, 0);
        matmul_transposed_both::launch(
            config,
            ctx,
            mod_,
            &d_a,
            &d_b,
            &mut d_c,
            m as u32,
            n as u32,
            k as u32,
        )
        .unwrap();
        d_c.copy_to_host(&mut output).unwrap();
    });
    assert_close(&output, &expected, 1e-2);
}

#[test]
fn test_matmul_batched() {
    let (batch, m, n, k) = (2usize, 16usize, 16usize, 16usize);
    let a: Vec<f32> = (0..batch * m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i % 11) as f32 * 0.1).collect();
    // CPU reference: each batch independently
    let mut expected = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let a_slice = &a[bi * m * k..(bi + 1) * m * k];
        let b_slice = &b[bi * k * n..(bi + 1) * k * n];
        let batch_result = cpu_matmul(a_slice, b_slice, m, n, k);
        expected[bi * m * n..(bi + 1) * m * n].copy_from_slice(&batch_result);
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
        matmul_batched::launch(
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
