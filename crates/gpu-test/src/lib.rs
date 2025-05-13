#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

/// RUSTFLAGS="-Zcodegen-backend=`realpath ../target/debug/librustc_codegen_gpu.dylib`" cargo build
///
pub const M: usize = 1024;
pub const N: usize = 4096;
pub const K: usize = 512;
pub const BK: usize = 16; // Block size for shared memory

#[no_mangle]
#[gpu_codegen::kernel]
/// assume BK * BK == number of threads in a block x axis.
fn kernel_print(a: &[u8]) {
    gpu::add_mlir_string_attr("\"run\"");
    gpu::printf();
}

#[no_mangle]
#[gpu_codegen::kernel]
/// assume BK * BK == number of threads in a block x axis.
fn kernel(a: &[u8]) {
    let c = kernel2();
    let d = c;
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let thread_id_x = gpu::thread_id();
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let thread_id = gpu::global_thread_id();
    gpu::add_mlir_string_attr("\"x\"");
    gpu::printf();
}

#[no_mangle]
#[gpu_codegen::device]
/// assume BK * BK == number of threads in a block x axis.
fn kernel2() -> u32 {
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let thread_id_x = gpu::thread_id();
    thread_id_x as u32
}

#[no_mangle]
#[gpu_codegen::host]
/// assume BK * BK == number of threads in a block x axis.
pub fn host(a: &[u8; 10], b: &[u8;10], result: &mut [u8; 10]) -> u32 {
    gpu::scope(|s, grid| {
        // Split data to avoid concurrent write.
        let mut chunks: std::slice::ChunksMut<'_, u8> = result.chunks_mut(1);
        for block in grid {
            // Split the flattened result into chunks of size `chunk_size` using `chunks_mut`
            let chunk = chunks.next().unwrap();
            let mut chunk_row = chunk.chunks_mut(1);
            let a_s = gpu::Shared::<[u8; BK * BK]>::new();
            let b_s = gpu::Shared::<[u8; BK * BK]>::new();
            for thread in block {
                let c: &mut [u8] = chunk_row.next().unwrap();
                // Spawn a thread for each chunk
                thread.spawn(move || kernel(&a, &b, &mut c[0], a_s, b_s, BK));
            }
        }
    }); // Ensure t
}
