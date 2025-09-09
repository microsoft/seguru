// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn mutate_shared() {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    shared[0] = 0; //~ ERROR cannot assign to data in dereference of
}
