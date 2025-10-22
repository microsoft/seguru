// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu::kernel]
#[gpu_codegen::device]
pub fn test_violate_divergence_check() {
    if gpu::thread_id::<gpu::DimX>() % 2 == 0 {
        return;
    }
    gpu::sync_threads(); 
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry sync_