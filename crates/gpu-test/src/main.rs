use gpu::Shared;

/// RUSTFLAGS="-Zcodegen-backend=`realpath ../target/debug/librustc_codegen_gpu.dylib`" cargo build
///
pub const M: usize = 1024;
pub const N: usize = 4096;
pub const K: usize = 512;
pub const BK: usize = 16; // Block size for shared memory

/// assume BK * BK == number of threads in a block x axis.
fn kernel(
    a: &[u8],
    b: &[u8],
    c: &mut u8,
    a_s: Shared<[u8; BK * BK]>,
    b_s: Shared<[u8; BK * BK]>,
    bk: usize,
) {
    let thread = gpu::thread();
    let unique_col = thread.local_thread_id() % bk;
    let unique_row = thread.local_thread_id() / bk;
    for bkIdx in (0..K).step_by(BK) {
        let mut mut_a = a_s.write();
        let mut mut_b = b_s.write();
        // populate the SMEM caches
        mut_a[unique_row * BK + unique_col] = a[unique_row * BK + bkIdx * BK + unique_col];
        mut_b[unique_row * BK + unique_col] = b[unique_row * BK + bkIdx * BK + unique_col];
        let a = a_s.read();
        let a = b_s.read();
        for k in 0..BK {
            *c = *c + a[unique_row * M + k] * b[k * N + unique_col]; // Modify each value in the chunk
        }
    }
}

fn main() {
    let rows = 2; // Adjust to fit your example size
    let cols = 2;

    // Initialize the result as a mutable vector
    let a = [0; K * M];
    let b = [0; N * K];
    let mut result = [0; M * N];

    // Flatten the vector into a slice to use chunk_mut

    // Specify chunk size (this determines how many elements each thread will process)
    let chunk_size = 1;

    // Similar to thread::scope to ensure all threads live during the scope
    // in gpu::grid should be called for only once inside one scope.
    gpu::scope(|s, grid| {
        // Split data to avoid concurrent write.
        let mut chunks: std::slice::ChunksMut<'_, u8> = result.chunks_mut(1);
        for block in grid {
            // Split the flattened result into chunks of size `chunk_size` using `chunks_mut`
            let mut chunk = chunks.next().unwrap();
            let mut chunk_row = chunk.chunks_mut(1);
            let a_s = gpu::Shared::<[u8; BK * BK]>::new();
            let b_s = gpu::Shared::<[u8; BK * BK]>::new();
            for thread in block {
                let c: &mut [u8] = chunk_row.next().unwrap();
                // Spawn a thread for each chunk
                thread.spawn(move || kernel(&a, &b, &mut c[0], a_s, b_s, BK));
            }
        }
    }); // Ensure that all threads finish execution

    // Print the final result after modification
    println!("{:?}", result);
}
