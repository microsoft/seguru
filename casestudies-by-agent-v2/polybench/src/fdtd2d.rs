//! FDTD-2D: `TMAX` time steps of the 2D finite-difference time-domain update
//! of `ex`, `ey` and `hz` over an `NX x NY` grid.
//!
//! Three kernels per step, matching the data dependences of the original
//! (`ey` and `ex` both read `hz`, `hz` then reads the updated `ex`/`ey`).
//! Every store goes through a chunk built by all threads and is made
//! unconditional by folding the boundary case into the stored *value*, which
//! keeps the kernels free of divergent `chunk_mut`.

use gpu::*;

pub const BX: u32 = 32;
pub const BY: u32 = 8;

/// `ey[0][j] = fict`, `ey[i][j] -= 0.5 * (hz[i][j] - hz[i-1][j])` otherwise.
#[gpu::cuda_kernel]
pub fn fdtd_ey(ey: &mut [f32], hz: &[f32], ny: u32, fict: f32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let j = block_id::<DimX>() * BX + thread_id::<DimX>();
    let i = block_id::<DimY>() * BY + thread_id::<DimY>();

    let mut out =
        chunk_mut(ey, reshape_map!([1] | [32, gx, 8, gy] => layout: [i0, t0, t1, t2, t3]));
    let im = i.max(1) - 1;
    let v = out[0] - 0.5 * (hz[(i * ny + j) as usize] - hz[(im * ny + j) as usize]);
    out[0] = if i == 0 { fict } else { v };
}

/// `ex[i][j] -= 0.5 * (hz[i][j] - hz[i][j-1])` for `j > 0`.
#[gpu::cuda_kernel]
pub fn fdtd_ex(ex: &mut [f32], hz: &[f32], ny: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let j = block_id::<DimX>() * BX + thread_id::<DimX>();
    let i = block_id::<DimY>() * BY + thread_id::<DimY>();

    let mut out =
        chunk_mut(ex, reshape_map!([1] | [32, gx, 8, gy] => layout: [i0, t0, t1, t2, t3]));
    let jm = j.max(1) - 1;
    let v = out[0] - 0.5 * (hz[(i * ny + j) as usize] - hz[(i * ny + jm) as usize]);
    out[0] = if j == 0 { out[0] } else { v };
}

/// `hz[i][j] -= 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])`.
#[gpu::cuda_kernel]
pub fn fdtd_hz(hz: &mut [f32], ex: &[f32], ey: &[f32], nx: u32, ny: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let j = block_id::<DimX>() * BX + thread_id::<DimX>();
    let i = block_id::<DimY>() * BY + thread_id::<DimY>();

    let mut out =
        chunk_mut(hz, reshape_map!([1] | [32, gx, 8, gy] => layout: [i0, t0, t1, t2, t3]));
    let jp = (j + 1).min(ny - 1);
    let ip = (i + 1).min(nx - 1);
    let v = out[0]
        - 0.7
            * (ex[(i * ny + jp) as usize] - ex[(i * ny + j) as usize] + ey[(ip * ny + j) as usize]
                - ey[(i * ny + j) as usize]);
    out[0] = if i + 1 < nx && j + 1 < ny { v } else { out[0] };
}

/// CPU reference returning `(ex, ey, hz)`.
pub fn fdtd2d_cpu(
    ex: &[f32],
    ey: &[f32],
    hz: &[f32],
    fict: &[f32],
    nx: usize,
    ny: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut ex = ex.to_vec();
    let mut ey = ey.to_vec();
    let mut hz = hz.to_vec();
    for t in 0..fict.len() {
        for j in 0..ny {
            ey[j] = fict[t];
        }
        for i in 1..nx {
            for j in 0..ny {
                ey[i * ny + j] -= 0.5 * (hz[i * ny + j] - hz[(i - 1) * ny + j]);
            }
        }
        for i in 0..nx {
            for j in 1..ny {
                ex[i * ny + j] -= 0.5 * (hz[i * ny + j] - hz[i * ny + j - 1]);
            }
        }
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                hz[i * ny + j] -= 0.7
                    * (ex[i * ny + j + 1] - ex[i * ny + j] + ey[(i + 1) * ny + j]
                        - ey[i * ny + j]);
            }
        }
    }
    (ex, ey, hz)
}

/// `nx` must be a multiple of [`BY`] and `ny` a multiple of [`BX`].
pub fn fdtd2d_gpu(
    ex: &[f32],
    ey: &[f32],
    hz: &[f32],
    fict: &[f32],
    nx: usize,
    ny: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(nx % BY as usize == 0 && ny % BX as usize == 0);
    gpu_host::cuda_ctx(0, |ctx, m| {
        let mut dex = ctx.new_tensor_view(ex).unwrap();
        let mut dey = ctx.new_tensor_view(ey).unwrap();
        let mut dhz = ctx.new_tensor_view(hz).unwrap();
        let gx = (ny / BX as usize) as u32;
        let gy = (nx / BY as usize) as u32;
        for &f in fict.iter() {
            let cfg = gpu_host::gpu_config!(gx, gy, 1, @const BX, @const BY, 1, 0);
            fdtd_ey::launch(cfg, ctx, m, &mut dey, &dhz, ny as u32, f).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, @const BX, @const BY, 1, 0);
            fdtd_ex::launch(cfg, ctx, m, &mut dex, &dhz, ny as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, @const BX, @const BY, 1, 0);
            fdtd_hz::launch(cfg, ctx, m, &mut dhz, &dex, &dey, nx as u32, ny as u32).unwrap();
        }
        let mut hex = vec![0.0f32; nx * ny];
        let mut hey = vec![0.0f32; nx * ny];
        let mut hhz = vec![0.0f32; nx * ny];
        dex.copy_to_host(&mut hex).unwrap();
        dey.copy_to_host(&mut hey).unwrap();
        dhz.copy_to_host(&mut hhz).unwrap();
        (hex, hey, hhz)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn fdtd2d_matches_cpu() {
        let (nx, ny, tmax) = (512usize, 512usize, 20usize);
        let ex = seq(nx * ny, 91);
        let ey = seq(nx * ny, 92);
        let hz = seq(nx * ny, 93);
        let fict = seq(tmax, 94);
        let (wex, wey, whz) = fdtd2d_cpu(&ex, &ey, &hz, &fict, nx, ny);
        let (gex, gey, ghz) = fdtd2d_gpu(&ex, &ey, &hz, &fict, nx, ny);
        assert_close(&gex, &wex, 1e-4, "fdtd2d ex");
        assert_close(&gey, &wey, 1e-4, "fdtd2d ey");
        assert_close(&ghz, &whz, 1e-4, "fdtd2d hz");
    }
}
