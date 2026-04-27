//! Pointwise 1x1 Conv2d over [B,Cin,H,W] with weights [Cout,Cin,1,1].
use gpu::prelude::*;
use std::path::Path;
use std::time::Instant;

const TILE_M: u32 = 16;
const TILE_N: u32 = 16;
const TILE_K: u32 = 16;
const X_STRIDE: u32 = TILE_K + 1;
const W_STRIDE: u32 = TILE_N + 1;
const X_TILE: u32 = TILE_M * X_STRIDE;
const W_TILE: u32 = TILE_K * W_STRIDE;

fn checked_len(name: &str, dims: &[usize]) -> usize {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .unwrap_or_else(|| panic!("{name} exceeds usize range"))
}

fn checked_u32(name: &str, value: usize) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{name} exceeds u32 range: {value}"))
}

#[gpu::cuda_kernel]
pub fn pointwise_conv2d_tiled_kernel(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    B: u32,
    Cin: u32,
    H: u32,
    Wd: u32,
    Cout: u32,
) {
    let _ = B;
    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let bx = block_id::<DimX>();
    let by = block_id::<DimY>();
    let bz = block_id::<DimZ>();

    let m = H * Wd;
    let hw = bx * TILE_M + tx;
    let co = by * TILE_N + ty;

    let mut xs = gpu::GpuShared::<[f32; X_TILE as usize]>::zero();
    let mut ws = gpu::GpuShared::<[f32; W_TILE as usize]>::zero();

    let x_load_map = reshape_map!(
        [1] | [TILE_M, (TILE_K, X_STRIDE)]
        => layout: [i0, t1, t0]
    );
    let w_load_map = reshape_map!(
        [1] | [TILE_K, (TILE_N, W_STRIDE)]
        => layout: [i0, t1, t0]
    );
    let out_map = reshape_map!(
        [1] | [(TILE_M, TILE_M), grid_dim::<DimX>(), (TILE_N, TILE_N), grid_dim::<DimY>(), grid_dim::<DimZ>()]
        => layout: [i0, t0, t1, t2, t3, t4]
    );
    let mut out = chunk_mut(y, out_map);

    let mut acc = 0.0f32;
    let mut k0 = 0u32;
    while k0 < Cin {
        let x_ci = k0 + ty;
        let w_ci = k0 + tx;
        {
            let mut x_chunk = xs.chunk_mut(x_load_map);
            x_chunk[0] = if hw < m && x_ci < Cin {
                x[((bz * Cin + x_ci) * m + hw) as usize]
            } else {
                0.0
            };
        }
        {
            let mut w_chunk = ws.chunk_mut(w_load_map);
            w_chunk[0] = if w_ci < Cin && co < Cout {
                w[(co * Cin + w_ci) as usize]
            } else {
                0.0
            };
        }
        sync_threads();

        let mut kk = 0u32;
        while kk < TILE_K {
            acc += xs[(tx * X_STRIDE + kk) as usize] * ws[(kk * W_STRIDE + ty) as usize];
            kk += 1;
        }
        sync_threads();
        k0 += TILE_K;
    }

    if hw < m && co < Cout {
        out[0] = acc;
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path,
    out_dir: &Path,
    iters: usize,
    shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 5, "pointwise_conv2d: shape=[B,Cin,H,W,Cout]");
    let (b, cin, h, wd, cout) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    assert_eq!(cin, 64, "pointwise_conv2d benchmark keeps Cin=64");
    assert_eq!(cout, 128, "pointwise_conv2d benchmark keeps Cout=128");
    let x_len = checked_len("pointwise_conv2d input elements", &[b, cin, h, wd]);
    let w_len = checked_len("pointwise_conv2d weight elements", &[cout, cin]);
    let y_len = checked_len("pointwise_conv2d output elements", &[b, cout, h, wd]);
    checked_u32("pointwise_conv2d input elements", x_len);
    checked_u32("pointwise_conv2d weight elements", w_len);
    let total = checked_u32("pointwise_conv2d output elements", y_len);
    let _ = total;
    let spatial = checked_len("pointwise_conv2d spatial elements", &[h, wd]);
    assert_eq!(
        spatial % TILE_M as usize,
        0,
        "pointwise_conv2d H*W={} must be a multiple of TILE_M={}",
        spatial,
        TILE_M
    );
    assert_eq!(
        cout % TILE_N as usize,
        0,
        "pointwise_conv2d Cout={} must be a multiple of TILE_N={}",
        cout,
        TILE_N
    );

    let h_x = crate::read_bin(&in_dir.join("x.bin"), x_len);
    let h_w = crate::read_bin(&in_dir.join("w.bin"), w_len);
    let mut h_y = vec![0f32; y_len];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = checked_u32("pointwise_conv2d B", b);
    let ci = checked_u32("pointwise_conv2d Cin", cin);
    let hh = checked_u32("pointwise_conv2d H", h);
    let ww = checked_u32("pointwise_conv2d W", wd);
    let co = checked_u32("pointwise_conv2d Cout", cout);
    let spatial_u = checked_u32("pointwise_conv2d spatial elements", spatial);
    let grid_x = spatial_u.div_ceil(TILE_M);
    let grid_y = co.div_ceil(TILE_N);

    {
        let cfg = gpu_host::gpu_config!(grid_x, grid_y, bb, TILE_M, TILE_N, 1, 0);
        pointwise_conv2d_tiled_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, ci, hh, ww, co,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(grid_x, grid_y, bb, TILE_M, TILE_N, 1, 0);
        pointwise_conv2d_tiled_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, ci, hh, ww, co,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(grid_x, grid_y, bb, TILE_M, TILE_N, 1, 0);
        pointwise_conv2d_tiled_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, ci, hh, ww, co,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
