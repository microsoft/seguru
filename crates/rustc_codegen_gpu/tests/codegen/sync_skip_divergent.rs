// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu::attr(skip_divergence_check)]
#[gpu_codegen::device]
pub fn test_violate_divergence_check() {
    if gpu::thread_id::<gpu::DimX>() % 2 == 0 {
        // We should have ERROR Invalid use of diversed data in GPU code, but skip_divergence_check skips the check.
        gpu::sync_threads(); 
    }
}
// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .func sync_skip_divergent