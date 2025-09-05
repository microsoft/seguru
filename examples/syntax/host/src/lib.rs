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
    let h_a = (1u32..(1 + len as u32)).collect::<Vec<u32>>();
    let mut h_b = (5..(5 + len as u32)).collect::<Vec<u32>>();
    assert!(h_b.len() == len);
    let h_c = (10..(10 + len as u32)).collect::<Vec<u32>>();
    let mut h_f = vec![0.0f32; len];
    let h_g = (1..(1 + len)).map(|x| x as f32 * 1.1).collect::<Vec<f32>>();

    let d_a = ctx.new_gmem_with_len::<u32>(len, &h_a)?;
    let d_b = ctx.new_gmem_with_len::<u32>(len, &h_b)?;
    let d_c = ctx.new_gmem_with_len::<u32>(len, &h_c)?;
    let d_f = ctx.new_gmem_with_len::<f32>(len, &h_f)?;
    let d_g = ctx.new_gmem_with_len::<f32>(len, &h_g)?;
    let d_b_c = gpu::GlobalThreadChunk::new_from_host(d_b, gpu::Map2D::new(w));

    // Now do the kernel
    let config = gpu_host::gpu_config!(1, 1, 1, 1, 4, 1, 0);
    kernel_arith(config, ctx, m, d_a, d_b_c, d_c, d_f, w, d_g).expect("Kernel execution failed");
    d_b.copy_to_host(&mut h_b, len, ctx)?;
    d_f.copy_to_host(&mut h_f, len, ctx)?;

    for (i, bi) in h_b.iter().enumerate().take(4) {
        println!("b[{}] = {}", i, bi);
        assert!(*bi == 42 + i as u32 * 2);
    }

    for (i, fi) in h_f.iter().enumerate().take(4) {
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
