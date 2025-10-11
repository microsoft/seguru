mod host;

use gpu_host::cuda_ctx;
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
    let mut h_h = 0.0f32;

    let d_a = ctx.new_tensor_view::<[u32]>(&h_a)?;
    let mut d_b = ctx.new_tensor_view::<[u32]>(&h_b)?;
    let d_c = ctx.new_tensor_view::<[u32]>(&h_c)?;
    let mut d_f = ctx.new_tensor_view::<[f32]>(&h_f)?;
    let d_g = ctx.new_tensor_view::<[f32]>(&h_g)?;
    let d_b_c = gpu::GlobalThreadChunk::new_from_host(&mut d_b, gpu::Map2D::new(w as u32));
    let mut d_h = ctx.new_tensor_view(&h_h)?;

    // Now do the kernel
    const BDIM_Y: u32 = 4;
    let config = gpu_host::gpu_config!(1, 1, 1, 1, @const BDIM_Y, 1, 0);
    kernel_arith(
        config, ctx, m, &d_a, d_b_c, &d_c, &mut d_f, w, &d_g, &mut d_h,
    )
    .expect("Kernel execution failed");
    d_b.copy_to_host(&mut h_b)?;
    d_f.copy_to_host(&mut h_f)?;
    d_h.copy_to_host(&mut h_h)?;

    for (i, bi) in h_b.iter().enumerate().take(BDIM_Y as _) {
        println!("b[{}] = {}", i, bi);
        assert!(*bi == 42 + i as u32 * 2);
    }

    for (i, fi) in h_f.iter().enumerate().take(BDIM_Y as _) {
        println!("f[{}] = {}", i, fi);
        match i {
            0 => assert!(*fi == 13.391208),
            1 => assert!(*fi == 11.808496),
            2 => assert!(*fi == 10.842255),
            3 => assert!(*fi == 10.048399),
            _ => unreachable!(),
        }
    }
    let mut sum = 0.0f32;
    for i in 0..BDIM_Y {
        sum += h_g[i as usize];
    }
    assert!(h_h == sum, "{} != {}: {:?}", h_h, sum, h_g);
    Ok(())
}

pub fn test_oob1() {
    cuda_ctx(0, |ctx, m| {
        let mut out = ctx.new_tensor_view::<[f32]>(&[0.0; 16]).unwrap();
        let config = gpu_host::gpu_config!(1, 1, 1, 16, 1, 1, 0);
        host::oob1(config, ctx, m, 1.1, &mut out, 1)
    })
    .expect("Kernel execution failed");
}

pub fn test_oob2() {
    cuda_ctx(0, |ctx, m| {
        let mut out = ctx.new_tensor_view::<[f32]>(&[0.0; 16]).unwrap();
        let config = gpu_host::gpu_config!(1, 1, 1, 8, 1, 1, 0);
        host::oob2(config, ctx, m, 1.1, &mut out, 4)
    })
    .expect("Kernel execution failed");
}

pub fn test_oob3() {
    cuda_ctx(0, |ctx, m| {
        let mut out = ctx.new_tensor_view::<[f32]>(&[0.0; 16]).unwrap();
        let config = gpu_host::gpu_config!(1, 1, 1, 8, 1, 1, 0);
        host::oob3(config, ctx, m, 1.1, &mut out, 16)
    })
    .expect("Kernel execution failed");
}

pub fn test_oob_no_fails() {
    cuda_ctx(0, |ctx, m| {
        let mut out = ctx.new_tensor_view::<[f32]>(&[0.0; 16]).unwrap();
        let config = gpu_host::gpu_config!(1, 1, 1, 8, 1, 1, 0);
        host::oob_no_fails(config, ctx, m, 1.1, &mut out, 16)
    })
    .expect("Kernel execution failed");
}
