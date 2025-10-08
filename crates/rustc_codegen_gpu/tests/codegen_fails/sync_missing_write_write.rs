// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_shared_write_write(a: &[u8], b: &mut [u8]) {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_shared1 = shared.chunk_mut(gpu::MapLinear::new(1)); //~ ERROR The write needs a `sync_threads` called before other read/write
    chunk_shared1[0] = a[gpu::thread_id::<gpu::DimX>() as usize]; 
    let mut chunk_shared2 = shared.chunk_mut(gpu::MapLinear::new(2)); //~ NOTE need `sync_threads` before this read/write
    chunk_shared2[0] = 1; 
}

