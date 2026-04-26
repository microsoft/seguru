//! 32_Conv2d_Scaling_Min — Conv2d + scale + channel min.
//!
//! PyTorch reference:
//!     tmp = F.conv2d(x, W, b) * 2.0
//!     y = torch.min(tmp, dim=1, keepdim=True)[0]
//!
//! Shapes: x [64,64,256,256], W [128,64,3,3], b [128], y [64,1,254,254].

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::prelude::*;

const BLK: u32 = 256;
const KSZ: u32 = 3;

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
pub fn conv2d_scale_kernel(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    tmp: &mut [f32],
    B: u32,
    Cin: u32,
    H: u32,
    Wd: u32,
    Cout: u32,
    Ho: u32,
    Wo: u32,
) {
    let _ = B;
    let mut tmp_chunk = chunk_mut(tmp, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let total = B * Cout * Ho * Wo;

    if idx < total {
        let wo = idx % Wo;
        let mut t = idx / Wo;
        let ho = t % Ho;
        t /= Ho;
        let co = t % Cout;
        let bi = t / Cout;

        let mut acc = bias[co as usize];
        let mut ci: u32 = 0;
        while ci < Cin {
            let x_base = (bi * Cin + ci) * H * Wd + ho * Wd + wo;
            let w_base = (co * Cin + ci) * (KSZ * KSZ);

            let mut kh: u32 = 0;
            while kh < KSZ {
                let x_row = x_base + kh * Wd;
                let w_row = w_base + kh * KSZ;
                acc += x[x_row as usize] * w[w_row as usize];
                acc += x[(x_row + 1) as usize] * w[(w_row + 1) as usize];
                acc += x[(x_row + 2) as usize] * w[(w_row + 2) as usize];
                kh += 1;
            }
            ci += 1;
        }

        tmp_chunk[0] = acc * 2.0;
    }
}

#[gpu::cuda_kernel]
pub fn channel_min_kernel(tmp: &[f32], y: &mut [f32], B: u32, Cout: u32, Ho: u32, Wo: u32) {
    let _ = B;
    let warp = ThreadWarpTile::<32>;
    let wpb = warp.meta_group_size();
    let out_idx = block_id::<DimX>() * wpb + warp.subgroup_id();
    let lane = warp.thread_rank();
    let total = B * Ho * Wo;
    let mut y_chunk = chunk_mut(
        y,
        reshape_map!([1] | [(32, 1), wpb * grid_dim::<DimX>()] => layout: [i0, t0, t1]),
    );

    if out_idx < total {
        let wo = out_idx % Wo;
        let t = out_idx / Wo;
        let ho = t % Ho;
        let bi = t / Ho;

        let mut neg_min = f32::NEG_INFINITY;
        let mut co = lane;
        while co < Cout {
            let v = tmp[((bi * Cout + co) * Ho * Wo + ho * Wo + wo) as usize];
            neg_min = neg_min.max(-v);
            co += warp.size();
        }
        let reduced = -warp.redux(ReduxMax, neg_min);
        if lane == 0 {
            y_chunk[0] = reduced;
        }
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
    assert_eq!(
        shape.len(),
        7,
        "conv2d_scaling_min: shape=[B, Cin, H, W, Cout, Kh, Kw]"
    );
    let (b, cin, h, wd, cout, kh, kw) = (
        shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6],
    );
    assert_eq!(kh, 3, "kernel specialized to Kh=3");
    assert_eq!(kw, 3, "kernel specialized to Kw=3");
    let ho = h - kh + 1;
    let wo = wd - kw + 1;

    let h_x = crate::read_bin(&in_dir.join("x.bin"), b * cin * h * wd);
    let h_w = crate::read_bin(&in_dir.join("W.bin"), cout * cin * kh * kw);
    let h_b = crate::read_bin(&in_dir.join("b.bin"), cout);
    let mut h_tmp = vec![0f32; b * cout * ho * wo];
    let mut h_y = vec![0f32; b * ho * wo];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
    let mut d_tmp = ctx.new_tensor_view(h_tmp.as_mut_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = b as u32;
    let cinn = cin as u32;
    let hh = h as u32;
    let ww = wd as u32;
    let co_u = cout as u32;
    let hou = ho as u32;
    let wou = wo as u32;
    let conv_total = bb * co_u * hou * wou;
    let out_total = bb * hou * wou;
    let conv_grid = conv_total.div_ceil(BLK);
    let min_grid = out_total.div_ceil(8);

    {
        let conv_cfg = gpu_host::gpu_config!(conv_grid, 1, 1, BLK, 1, 1, 0);
        conv2d_scale_kernel::launch(
            conv_cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_tmp, bb, cinn, hh, ww, co_u, hou, wou,
        )
        .unwrap();
        let min_cfg = gpu_host::gpu_config!(min_grid, 1, 1, BLK, 1, 1, 0);
        channel_min_kernel::launch(min_cfg, ctx, md, &d_tmp, &mut d_y, bb, co_u, hou, wou).unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let conv_cfg = gpu_host::gpu_config!(conv_grid, 1, 1, BLK, 1, 1, 0);
        conv2d_scale_kernel::launch(
            conv_cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_tmp, bb, cinn, hh, ww, co_u, hou, wou,
        )
        .unwrap();
        let min_cfg = gpu_host::gpu_config!(min_grid, 1, 1, BLK, 1, 1, 0);
        channel_min_kernel::launch(min_cfg, ctx, md, &d_tmp, &mut d_y, bb, co_u, hou, wou).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let conv_cfg = gpu_host::gpu_config!(conv_grid, 1, 1, BLK, 1, 1, 0);
        conv2d_scale_kernel::launch(
            conv_cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_tmp, bb, cinn, hh, ww, co_u, hou, wou,
        )
        .unwrap();
        let min_cfg = gpu_host::gpu_config!(min_grid, 1, 1, BLK, 1, 1, 0);
        channel_min_kernel::launch(min_cfg, ctx, md, &d_tmp, &mut d_y, bb, co_u, hou, wou).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_tmp);
    drop(d_b);
    drop(d_w);
    drop(d_x);

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
