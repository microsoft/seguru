#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;
use core::marker::PhantomData;

#[gpu_codegen::device]
#[no_mangle]
/// assume BK * BK == number of threads in a block x axis.
fn kernel(a: &[u8], b: &mut u8) -> u8 {
    let thread_id_x = gpu::thread_id(gpu::DimType::X);
    *b = a[thread_id_x];
    *b
}

pub struct ThreadScope<'scope, 'env: 'scope> {
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
    val: u32,
}

#[gpu_codegen::device]
#[inline(never)]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope ThreadScope<'scope, 'env>) -> T + Send,
{
    f(&ThreadScope { scope: PhantomData, env: PhantomData, val: 0 })
}



#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn kernel_arith(a: &[u8], b: &mut [u8], window: usize) {
    let mut b = gpu::GlobalThreadChunk::new(b, gpu::MapLinear::new(window));
    let c = &mut b[0];
    let val = scope(|s| {
        kernel(a, c)
    });
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry thread_scope_3A__3A_kernel_arith
// PTX_CHECK: [thread_scope_3A__3A_kernel_arith_param_0];
// PTX_CHECK: [thread_scope_3A__3A_kernel_arith_param_1];
// PTX_CHECK: [thread_scope_3A__3A_kernel_arith_param_2];
// PTX_CHECK: [thread_scope_3A__3A_kernel_arith_param_4];
// PTX_CHECK: tid.x;
// PTX_CHECK: tid.y;
// PTX_CHECK: tid.z;
