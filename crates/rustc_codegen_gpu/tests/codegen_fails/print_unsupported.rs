// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub fn print_fails(a: &[u8], b: &[u8]) {
    let id = gpu::thread_id::<gpu::DimX>();
    gpu::println!("{} {} at {:x}", a[0], b[0] as u32, id); //~ ERROR Currently we only support '{}' as placeholder
}