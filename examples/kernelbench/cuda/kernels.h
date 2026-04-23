#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Elementwise (11)
float bench_relu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_leaky_relu_forward(const float* input, float* output, int n, float alpha, int grid, int block, int warmup, int iters);
float bench_sigmoid_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_tanh_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_swish_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_selu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_hard_sigmoid_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_softplus_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_softsign_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_elu_forward(const float* input, float* output, int n, float alpha, int grid, int block, int warmup, int iters);
float bench_hard_tanh_forward(const float* input, float* output, int n, float min_val, float max_val, int grid, int block, int warmup, int iters);

// GELU (2)
float bench_gelu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_mingpt_new_gelu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);

// Matmul 2D (4)
float bench_matmul_forward(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);
float bench_matmul_transposed_a(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);
float bench_matmul_transposed_b(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);
float bench_matmul_transposed_both(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);

// Matmul batched (2)
float bench_matmul_batched(const float* a, const float* b, float* c, int batch, int m, int n, int k, int grid, int block, int warmup, int iters);
float bench_tensor3d_matmul(const float* a, const float* b, float* c, int batch, int m, int n, int k, int grid, int block, int warmup, int iters);

// Matvec (3)
float bench_matvec_forward(const float* a, const float* x, float* y, int m, int n, int grid, int block, int warmup, int iters);
float bench_scalar_multiply(const float* input, float* output, float s, int n, int grid, int block, int warmup, int iters);
float bench_tensor3d_matvec(const float* a, const float* b, float* c, int batch, int m, int n, int k, int grid, int block, int warmup, int iters);

// Reduction (4)
float bench_sum_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_mean_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_max_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_min_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);

// Argreduce (2)
float bench_argmax_reduce(const float* input, unsigned int* output, int batch, int dim, int block, int warmup, int iters);
float bench_argmin_reduce(const float* input, unsigned int* output, int batch, int dim, int block, int warmup, int iters);

// Softmax (2)
float bench_softmax_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_log_softmax_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);

// Norm (5)
float bench_rms_norm_forward(const float* input, float* output, int batch, int dim, float eps, int block, int warmup, int iters);
float bench_frobenius_norm_forward(const float* input, float* output, int n, int block, int warmup, int iters);
float bench_l1_norm_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_l2_norm_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_layer_norm_forward(const float* input, float* output, int batch, int dim, float eps, int block, int warmup, int iters);

// Loss (4)
float bench_mse_loss_forward(const float* predictions, const float* targets, float* output, int n, int block, int warmup, int iters);
float bench_huber_loss_forward(const float* predictions, const float* targets, float* output, int n, float delta, int block, int warmup, int iters);
float bench_kl_div_loss_forward(const float* log_predictions, const float* targets, float* output, int n, int block, int warmup, int iters);
float bench_hinge_loss_forward(const float* predictions, const float* targets, float* output, int n, int block, int warmup, int iters);

// Cumulative (4)
float bench_cumsum_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);
float bench_cumprod_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);
float bench_cumsum_reverse_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);
float bench_cumsum_exclusive_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);

#ifdef __cplusplus
}
#endif
