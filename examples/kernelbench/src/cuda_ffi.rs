//! FFI bindings to CUDA reference benchmark kernels.
//! Used by the bench binary, not included in lib.rs (which is no_std).

extern "C" {
    // Elementwise (11)
    pub fn bench_relu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_leaky_relu_forward(input: *const f32, output: *mut f32, n: i32, alpha: f32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_sigmoid_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_tanh_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_swish_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_selu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_hard_sigmoid_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_softplus_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_softsign_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_elu_forward(input: *const f32, output: *mut f32, n: i32, alpha: f32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_hard_tanh_forward(input: *const f32, output: *mut f32, n: i32, min_val: f32, max_val: f32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // GELU (2)
    pub fn bench_gelu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_mingpt_new_gelu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Matmul 2D (4)
    pub fn bench_matmul_forward(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_matmul_transposed_a(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_matmul_transposed_b(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_matmul_transposed_both(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;

    // Matmul batched (2)
    pub fn bench_matmul_batched(a: *const f32, b: *const f32, c: *mut f32, batch: i32, m: i32, n: i32, k: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_tensor3d_matmul(a: *const f32, b: *const f32, c: *mut f32, batch: i32, m: i32, n: i32, k: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Matvec (3)
    pub fn bench_matvec_forward(a: *const f32, x: *const f32, y: *mut f32, m: i32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_scalar_multiply(input: *const f32, output: *mut f32, s: f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_tensor3d_matvec(a: *const f32, b: *const f32, c: *mut f32, batch: i32, m: i32, n: i32, k: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Reduction (4)
    pub fn bench_sum_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_mean_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_max_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_min_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Argreduce (2)
    pub fn bench_argmax_reduce(input: *const f32, output: *mut u32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_argmin_reduce(input: *const f32, output: *mut u32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Softmax (2)
    pub fn bench_softmax_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_log_softmax_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Norm (5)
    pub fn bench_rms_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, eps: f32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_frobenius_norm_forward(input: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_l1_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_l2_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_layer_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, eps: f32, block: i32, warmup: i32, iters: i32) -> f32;

    // Loss (4)
    pub fn bench_mse_loss_forward(predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_huber_loss_forward(predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, delta: f32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_kl_div_loss_forward(log_predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_hinge_loss_forward(predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Cumulative (4)
    pub fn bench_cumsum_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_cumprod_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_cumsum_reverse_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_cumsum_exclusive_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
}
