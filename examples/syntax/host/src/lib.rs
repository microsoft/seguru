mod host;

use host::kernel_arith;

pub fn run_host_arith<'ctx>(
    ctx: &gpu_host::GpuCtxZeroGuard<'ctx, '_>,
    m: &'ctx gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    len: usize,
    w: usize,
) -> Result<(), gpu_host::CudaError> {
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
    let config = gpu_host::gpu_config!(1, 1, 1, 1, 4, 1, 0);
    kernel_arith(config, ctx, m, d_a_c, d_b_c, d_c, d_f_c, d_g).expect("Kernel execution failed");
    d_b.copy_to_host(h_b, min(h_b.len(), len), ctx)?;
    d_f.copy_to_host(h_f, min(h_f.len(), len), ctx)?;

    for (i, bi) in h_b.iter().enumerate().take(len) {
        println!("b[{}] = {}", i, bi);
        assert!(*bi == 42 + i as u32 * 2);
    }

    for (i, fi) in h_f.iter().enumerate().take(len) {
        println!("f[{}] = {}", i, fi);
        match i {
            0 => assert!(*fi == 13.391208),
            1 => assert!(*fi == 11.808496),
            2 => assert!(*fi == 10.842255),
            3 => assert!(*fi == 10.048399),
            _ => unreachable!(),
        }
    }
    Ok(())
}
