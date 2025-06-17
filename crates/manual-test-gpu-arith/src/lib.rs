#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_codegen::device]
#[inline(always)]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    b[0] = a[0];
}

#[no_mangle]
#[gpu_codegen::kernel]
pub fn kernel_arith_wrapper(a: &[u8], a_window: usize, b: &mut [u8], b_window: usize) {
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let c = gpu::thread_id() as usize;
    
    let a_local: &[u8];
    let mut b_local: &mut [u8];

    unsafe {
        // a_local = &a[(c * a_window)..(c * a_window + a_window)] as &[u8];
        // b_local = &mut b[(c * b_window)..(c * a_window + b_window)];

        // Use unsafe ptr conversion since the default subslice stuff is just plain pain to use
        let a_raw_ptr = a.as_ptr();
        let a_slice_start = a_raw_ptr.add(c * a_window);
        //a_local = &*core::intrinsics::aggregate_raw_ptr::<?Sized, *const u8, usize>(a_slice_start, a_window);
        a_local = core::slice::from_raw_parts(a_slice_start, a_window);

        
        let b_raw_ptr = b.as_mut_ptr();
        let b_slice_start = b_raw_ptr.add(c * b_window);
        //b_local = &mut *core::intrinsics::aggregate_raw_ptr(b_slice_start, b_window);
        b_local = core::slice::from_raw_parts_mut(b_slice_start, b_window);
    }

    kernel_arith(a_local, b_local);
}


