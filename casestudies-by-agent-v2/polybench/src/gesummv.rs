//! GESUMMV: `y = alpha * A x + beta * B x`, with `A`, `B` of shape `N x N`.
//!
//! Both products reduce along a row, so a single kernel walks the two rows
//! together: one warp per output element, `Float4` loads, and a pair of
//! shuffle reductions at the end. Fusing the two products halves the launch
//! overhead and lets the `x` vector be loaded once for both matrices.

use gpu::*;

use crate::warp_sum;

pub const MV_BX: u32 = 32;
pub const MV_BY: u32 = 8;

#[gpu::cuda_kernel]
pub fn gesummv_kernel(
    a: &[Float4],
    b: &[Float4],
    x: &[Float4],
    y: &mut [f32],
    n4: u32,
    alpha: f32,
    beta: f32,
) {
    assert!(Config::BDIM_X == MV_BX);
    assert!(Config::BDIM_Y == MV_BY);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gy = grid_dim::<DimY>();
    let base = (block_id::<DimY>() * MV_BY + ty) * n4;

    let mut sa = 0.0f32;
    let mut sb = 0.0f32;
    let mut j = tx;
    while j < n4 {
        let av = a[(base + j) as usize];
        let bv = b[(base + j) as usize];
        let xv = x[j as usize];
        sa += av[0] * xv[0] + av[1] * xv[1] + av[2] * xv[2] + av[3] * xv[3];
        sb += bv[0] * xv[0] + bv[1] * xv[1] + bv[2] * xv[2] + bv[3] * xv[3];
        j += MV_BX;
    }
    let sa = warp_sum(sa);
    let sb = warp_sum(sb);

    let mut out = chunk_mut(
        y,
        reshape_map!([1] | [(32, 1), 1, 8, gy] => layout: [i0, t0, t1, t2, t3]),
    );
    if tx == 0 {
        out[0] = alpha * sa + beta * sb;
    }
}

/// CPU reference.
pub fn gesummv_cpu(
    a: &[f32],
    b: &[f32],
    x: &[f32],
    n: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sa = 0.0f32;
        let mut sb = 0.0f32;
        for j in 0..n {
            sa += a[i * n + j] * x[j];
            sb += b[i * n + j] * x[j];
        }
        y[i] = alpha * sa + beta * sb;
    }
    y
}

pub fn gesummv_gpu(
    a: &[f32],
    b: &[f32],
    x: &[f32],
    n: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let pn = crate::common::round_up(n, (MV_BX * 4).max(MV_BY) as usize);
    let ha = crate::common::to_float4(&crate::common::pad2(a, n, n, pn, pn));
    let hb = crate::common::to_float4(&crate::common::pad2(b, n, n, pn, pn));
    let hx = crate::common::to_float4(&crate::common::pad1(x, pn));

    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let db = ctx.new_tensor_view(hb.as_slice()).unwrap();
        let dx = ctx.new_tensor_view(hx.as_slice()).unwrap();
        let zy = vec![0.0f32; pn];
        let mut dy = ctx.new_tensor_view(zy.as_slice()).unwrap();

        let cfg = gpu_host::gpu_config!(
            1, (pn / MV_BY as usize) as u32, 1, @const MV_BX, @const MV_BY, 1, 0);
        gesummv_kernel::launch(cfg, ctx, m, &da, &db, &dx, &mut dy, (pn / 4) as u32, alpha, beta)
            .unwrap();

        let mut h = vec![0.0f32; pn];
        dy.copy_to_host(&mut h).unwrap();
        h
    });
    out[..n].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn gesummv_matches_cpu() {
        let n = 1024usize;
        let a = seq(n * n, 41);
        let b = seq(n * n, 42);
        let x = seq(n, 43);
        let (alpha, beta) = (1.5f32, -0.75f32);
        let want = gesummv_cpu(&a, &b, &x, n, alpha, beta);
        let got = gesummv_gpu(&a, &b, &x, n, alpha, beta);
        assert_close(&got, &want, 1e-4, "gesummv");
    }
}
