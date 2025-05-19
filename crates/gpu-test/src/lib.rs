#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
//#![no_std]

use gpu::ThreadScope;

/// RUSTFLAGS="-Zcodegen-backend=`realpath ../target/debug/librustc_codegen_gpu.dylib`" cargo build
///
pub const M: usize = 1024;
pub const N: usize = 4096;
pub const K: usize = 512;
pub const BK: usize = 16; // Block size for shared memory

#[no_mangle]
#[gpu_codegen::kernel]
/// assume BK * BK == number of threads in a block x axis.
fn kernel_print() {
    gpu::add_mlir_string_attr("\"run\"");
    gpu::printf();
}

#[no_mangle]
#[gpu_codegen::device]
/// assume BK * BK == number of threads in a block x axis.
fn kernel(a: &[u8]) {
    let c = kernel2(a);
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
fn kernel2(b: &[u8]) -> u32 {
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let thread_id_x = gpu::thread_id();
    thread_id_x as u32
}

#[no_mangle]
#[gpu_codegen::host]
/// assume BK * BK == number of threads in a block x axis.
pub fn host(a: &mut [u8; 10], b: &[u8; 10], c: &mut [u8; 10]) {
    a[0] = 1;
    gpu::scope(|s| s.launch([1, 1, 1], [2, 2, 2], || kernel_print()));
    gpu::scope(|s| s.launch([1, 1, 1], [2, 2, 2], || kernel_print()));
    gpu::scope(
        #[gpu_codegen::device]
        |s| {
            for x in gpu::grid(4, 1, 1) {
                s.launch([1, 1, 1], [2, 2, 2], || kernel(a));
            }
            // Split data to avoid concurrent write.
            //for x in gpu::grid(4, 1, 1) {
            // Split the flattened result into chunks of size `chunk_size` using `chunks_mut`
            //for y in gpu::block(4, 1, 1) {
            // Spawn a thread for each chunk
            //gpu::launch(*x, *y, move || kernel(a));
            //}
            //}
        },
    ); // Ensure t
    kernel(a);
}
