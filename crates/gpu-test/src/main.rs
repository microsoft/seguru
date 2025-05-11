#![feature(register_tool)]
#![register_tool(gpu_codegen)]

/// RUSTFLAGS="-Zcodegen-backend=`realpath ../target/debug/librustc_codegen_gpu.dylib`" cargo build
///
pub const M: usize = 1024;
pub const N: usize = 4096;
pub const K: usize = 512;
pub const BK: usize = 16; // Block size for shared memory


#[no_mangle]
#[gpu_codegen::kernel]
/// assume BK * BK == number of threads in a block x axis.
fn kernel(a: &[u8]) {
    gpu::add_mlir_string_attr("\"x\"");
    let thread_id_x = gpu::thread_id();
    gpu::add_mlir_string_attr("\"x\"");
    let thread_id = gpu::global_thread_id();
    gpu::add_mlir_string_attr("\"x\"");
    gpu::printf();
    //println!("thread: {} {} {} {}", 1, thread_id_x, thread_id, a[0]);
}

#[no_mangle]
#[gpu_codegen::kernel]
/// assume BK * BK == number of threads in a block x axis.
fn kernel2() {
    gpu::add_mlir_string_attr("\"x\"");
    let thread_id_x = gpu::thread_id();
}

fn main() {
    // Initialize the result as a mutable vector
    let a = [0; 1];
    gpu::scope(|s, grid| {
        // Split data to avoid concurrent write.
        for block in grid {
            for thread in block {
                // Spawn a thread for each chunk
                thread.spawn(move || kernel(&a));
            }
        }
    }); // Ensure that all threads finish execution
}
