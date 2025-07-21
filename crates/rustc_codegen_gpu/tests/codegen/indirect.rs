#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

use core::marker::PhantomData;
#[repr(packed, C)]
pub struct A {
    a: i32,
    b: u32,
    c: u64,
    d: f32,
}

#[repr(packed, C)]
pub struct C<'a> {
    slice: &'a mut [u32],
    window: usize,
    dummy: PhantomData<&'a mut u32>,
}

#[repr(packed, C)]
pub struct C2<'a> {
    slice: &'a [u32],
    window: usize,
    dummy: PhantomData<&'a mut u32>,
}

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn assign_struct(a: A, b: C<'_>, c: C2<'_>) { //~ ERROR Kernel entry does not support fn abi indirect
    let thread_id = gpu::thread_id(gpu::DimType::Y);
    b.slice[gpu::thread_id(gpu::DimType::X)] = c.slice[gpu::thread_id(gpu::DimType::X)];
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry assign_struct
// PTX_CHECK: ld.global.u32
// PTX_CHECK: st.global.u32