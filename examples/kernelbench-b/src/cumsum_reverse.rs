//! Reverse row-wise cumulative sum: cumsum(x.flip(1), dim=1).flip(1).
use gpu::prelude::*;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 256;

#[gpu::cuda_kernel]
pub fn cumsum_reverse_kernel(x: &[f32], y: &mut [f32], d: u32) {
    let tid = thread_id::<DimX>();
    let row = block_id::<DimX>() as usize;
    let d_usize = d as usize;
    let x_row = &x[(row * d_usize)..((row + 1) * d_usize)];
    let items = d / block_dim::<DimX>();
    let base = (tid * items) as usize;
    let mut smem = GpuShared::<[f32; 256]>::zero();
    let out_map = reshape_map!(
        [d / block_dim::<DimX>()] | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [i0, t0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);

    let mut run = 0.0f32;
    let mut j = items;
    while j > 0 {
        j -= 1;
        run += x_row[base + j as usize];
        y_chunk[j] = run;
    }

    {
        let mut s = smem.chunk_mut(MapContinuousLinear::new(1));
        s[0] = run;
    }
    sync_threads();
    let mut off = 1u32;
    while off < BLOCK {
        let v = if tid + off < BLOCK {
            smem[(tid + off) as usize]
        } else {
            0.0
        };
        sync_threads();
        let nv = smem[tid as usize] + v;
        {
            let mut s = smem.chunk_mut(MapContinuousLinear::new(1));
            s[0] = nv;
        }
        sync_threads();
        off <<= 1;
    }
    let offset = if tid + 1 < BLOCK {
        smem[(tid + 1) as usize]
    } else {
        0.0
    };

    let mut j = 0u32;
    while j < items {
        y_chunk[j] += offset;
        j += 1;
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
    assert_eq!(shape.len(), 2);
    let (b, d) = (shape[0], shape[1]);
    assert_eq!(d % BLOCK as usize, 0, "D must be divisible by BLOCK");
    let n = b * d;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let dd = d as u32;
    let bs = BLOCK;
    let gs = b as u32;
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        cumsum_reverse_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        cumsum_reverse_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        cumsum_reverse_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
