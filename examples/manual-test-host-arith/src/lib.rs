mod host;

use host::kernel_arith;

use cuda_bindings::load_module_from_extern;

pub fn run_host_arith(
    ctx: &cuda_bindings::GpuCtxZeroGuard<'_, '_>,
    len: usize,
    w: usize,
) -> Result<(), cuda_bindings::CudaError> {
    let m = unsafe { load_module_from_extern!(ctx, gpu_bin_cst)? };
    // Allocate the two host-side arrays
    // TODO: Make h_a non-mutable
    let h_a: &[u32] = &[1, 2, 3, 4];
    let h_b: &mut [u32] = &mut [5, 6, 7, 8];
    let h_c: &[u32] = &[10, 11, 12, 13];
    let h_f: &mut [f32] = &mut [0.0, 0.0, 0.0, 0.0];
    let h_g: &[f32] = &[1.1, 2.2, 3.3, 4.4];

    let d_a = ctx.new_gmem_with_len::<u32>(len)?;
    let d_b = ctx.new_gmem_with_len::<u32>(len)?;
    let d_c = ctx.new_gmem_with_len::<u32>(len)?;
    let d_f = ctx.new_gmem_with_len::<f32>(len)?;
    let d_g = ctx.new_gmem_with_len::<f32>(len)?;
    use std::cmp::min;
    d_a.copy_from_host(h_a, min(h_a.len(), len), ctx)?;
    d_b.copy_from_host(h_b, min(h_b.len(), len), ctx)?;
    d_c.copy_from_host(h_c, min(h_c.len(), len), ctx)?;
    d_f.copy_from_host(h_f, min(h_f.len(), len), ctx)?;
    d_g.copy_from_host(h_g, min(h_g.len(), len), ctx)?;

    let d_a_c = gpu::GpuChunkable2D::<u32>::new(d_a, 1);
    let d_b_c = gpu::GpuChunkableMut2D::<u32>::new(d_b, 1);

    let d_f_c = gpu::GpuChunkableMut::<f32>::new(d_f, w);

    // Now do the kernel
    let config = cuda_bindings::GPUConfig {
        grid_dim_x: 1,
        grid_dim_y: 1,
        grid_dim_z: 1,
        block_dim_x: 1,
        block_dim_y: 4,
        block_dim_z: 1,
    };
    kernel_arith(ctx, m, config, d_a_c, d_b_c, d_c, d_f_c, d_g).expect("Kernel execution failed");
    d_b.copy_to_host(h_b, min(h_b.len(), len), ctx)?;
    d_f.copy_to_host(h_f, min(h_f.len(), len), ctx)?;

    for (i, bi) in h_b.iter().enumerate().take(len) {
        println!("b[{}] = {}", i, bi);
    }

    for (i, fi) in h_f.iter().enumerate().take(len) {
        println!("f[{}] = {}", i, fi);
    }
    Ok(())
}
