// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[allow(non_upper_case_globals)]
#[gpu_codegen::shared_size]
pub static shared_size_alloc_shared: usize = 9; //~ ERROR Shared memory size mismatch: expected 9, found 10

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn alloc_shared(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize) {
    let mut shared: gpu::GpuShared::<[u8; 10]> = gpu::GpuShared::uninit();
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry alloc_shared
// PTX_CHECK: bar.sync
// PTX_CHECK: st.shared
// PTX_CHECK: ld.shared
