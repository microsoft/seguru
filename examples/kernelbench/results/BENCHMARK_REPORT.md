# KernelBench Level 1 Performance Report

## Summary

| Metric | Value |
|---|---|
| Total kernels benchmarked | 42 |
| Total measurements | 84 |
| SeGuRu avg overhead vs CUDA | **2.02×** |
| SeGuRu avg overhead vs PyTorch | **2.93×** |
| Best (closest to CUDA) | matmul_transposed_b (large) — 0.99× |
| Worst (farthest from CUDA) | matmul_transposed_a (large) — 3.88× |

## Overhead by Input Size

| Size | Avg SeGuRu/CUDA |
|---|---|
| small | 2.01× |
| large | 2.02× |

## Results by Category

### Argreduce (2 kernels)

Average SeGuRu/CUDA ratio: **2.34×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| argmax_reduce | small | 12.51 | 6.14 | 26.62 | 2.04× | 0.47× |
| argmin_reduce | small | 12.23 | 6.14 | 26.62 | 1.99× | 0.46× |
| argmax_reduce | large | 29.39 | 11.26 | 34.82 | 2.61× | 0.84× |
| argmin_reduce | large | 30.63 | 11.26 | 34.82 | 2.72× | 0.88× |

### Batched_matmul (2 kernels)

Average SeGuRu/CUDA ratio: **3.01×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| matmul_batched | small | 15.44 | 6.66 | 31.74 | 2.32× | 0.49× |
| tensor3d_matmul | small | 15.60 | 7.17 | 31.74 | 2.18× | 0.49× |
| matmul_batched | large | 636.32 | 168.96 | 70.66 | 3.77× | 9.01× |
| tensor3d_matmul | large | 636.07 | 168.96 | 70.66 | 3.76× | 9.00× |

### Cumulative (4 kernels)

Average SeGuRu/CUDA ratio: **1.49×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| cumsum_forward | small | 25.58 | 17.41 | 21.50 | 1.47× | 1.19× |
| cumprod_forward | small | 25.47 | 17.41 | 21.50 | 1.46× | 1.18× |
| cumsum_reverse_forward | small | 27.29 | 17.41 | 45.06 | 1.57× | 0.61× |
| cumsum_exclusive_forward | small | 26.82 | 18.43 | 63.49 | 1.46× | 0.42× |
| cumsum_forward | large | 394.53 | 264.70 | 128.00 | 1.49× | 3.08× |
| cumprod_forward | large | 417.78 | 280.58 | 128.00 | 1.49× | 3.26× |
| cumsum_reverse_forward | large | 438.40 | 285.70 | 179.20 | 1.53× | 2.45× |
| cumsum_exclusive_forward | large | 430.01 | 292.35 | 167.94 | 1.47× | 2.56× |

### Elementwise (11 kernels)

Average SeGuRu/CUDA ratio: **1.87×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| relu_forward | small | 10.55 | 5.12 | 26.62 | 2.06× | 0.40× |
| sigmoid_forward | small | 10.87 | 5.12 | 21.50 | 2.12× | 0.51× |
| tanh_forward | small | 10.67 | 5.12 | 21.50 | 2.08× | 0.50× |
| swish_forward | small | 10.69 | 5.12 | 23.55 | 2.09× | 0.45× |
| selu_forward | small | 10.64 | 5.12 | 24.58 | 2.08× | 0.43× |
| hard_sigmoid_forward | small | 10.77 | 5.12 | 22.53 | 2.10× | 0.48× |
| softplus_forward | small | 10.67 | 5.12 | 22.53 | 2.08× | 0.47× |
| softsign_forward | small | 10.76 | 5.12 | 49.15 | 2.10× | 0.22× |
| leaky_relu_forward | small | 10.84 | 5.12 | 23.55 | 2.12× | 0.46× |
| elu_forward | small | 10.70 | 5.12 | 24.58 | 2.09× | 0.44× |
| hard_tanh_forward | small | 10.77 | 5.12 | 28.67 | 2.10× | 0.38× |
| relu_forward | large | 14.02 | 8.19 | 21.50 | 1.71× | 0.65× |
| sigmoid_forward | large | 14.20 | 8.19 | 21.50 | 1.73× | 0.66× |
| tanh_forward | large | 14.17 | 8.19 | 20.48 | 1.73× | 0.69× |
| swish_forward | large | 14.27 | 10.24 | 21.50 | 1.39× | 0.66× |
| selu_forward | large | 13.45 | 8.19 | 22.53 | 1.64× | 0.60× |
| hard_sigmoid_forward | large | 14.38 | 8.19 | 20.48 | 1.76× | 0.70× |
| softplus_forward | large | 14.19 | 9.22 | 21.50 | 1.54× | 0.66× |
| softsign_forward | large | 14.30 | 9.73 | 48.13 | 1.47× | 0.30× |
| leaky_relu_forward | large | 14.24 | 8.19 | 23.55 | 1.74× | 0.60× |
| elu_forward | large | 14.07 | 8.19 | 30.72 | 1.72× | 0.46× |
| hard_tanh_forward | large | 14.17 | 8.19 | 26.62 | 1.73× | 0.53× |

### Gelu (2 kernels)

Average SeGuRu/CUDA ratio: **1.89×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| gelu_forward | small | 10.76 | 5.12 | 22.53 | 2.10× | 0.48× |
| mingpt_new_gelu_forward | small | 11.02 | 5.12 | 21.50 | 2.15× | 0.51× |
| gelu_forward | large | 14.63 | 8.70 | 20.48 | 1.68× | 0.71× |
| mingpt_new_gelu_forward | large | 15.04 | 9.22 | 21.50 | 1.63× | 0.70× |

### Loss (4 kernels)

Average SeGuRu/CUDA ratio: **2.52×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| mse_loss_forward | small | 14.43 | 6.14 | 52.22 | 2.35× | 0.28× |
| huber_loss_forward | small | 14.71 | 6.14 | 51.20 | 2.40× | 0.29× |
| kl_div_loss_forward | small | 17.47 | 9.22 | 70.66 | 1.89× | 0.25× |
| hinge_loss_forward | small | 14.45 | 6.14 | 79.87 | 2.35× | 0.18× |
| mse_loss_forward | large | 798.22 | 231.42 | 55.30 | 3.45× | 14.43× |
| huber_loss_forward | large | 850.53 | 308.22 | 55.30 | 2.76× | 15.38× |
| kl_div_loss_forward | large | 1606.12 | 974.34 | 75.78 | 1.65× | 21.19× |
| hinge_loss_forward | large | 798.79 | 240.64 | 88.06 | 3.32× | 9.07× |

### Matmul_2d (4 kernels)

Average SeGuRu/CUDA ratio: **2.22×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| matmul_forward | small | 18.91 | 7.17 | 34.82 | 2.64× | 0.54× |
| matmul_transposed_a | small | 19.20 | 7.17 | 39.94 | 2.68× | 0.48× |
| matmul_transposed_b | small | 17.70 | 12.29 | 38.91 | 1.44× | 0.45× |
| matmul_transposed_both | small | 19.33 | 12.29 | 38.91 | 1.57× | 0.50× |
| matmul_forward | large | 2456.78 | 690.18 | 149.50 | 3.56× | 16.43× |
| matmul_transposed_a | large | 2618.94 | 674.30 | 158.72 | 3.88× | 16.50× |
| matmul_transposed_b | large | 3996.50 | 4018.18 | 155.65 | 0.99× | 25.68× |
| matmul_transposed_both | large | 3798.09 | 3787.78 | 157.70 | 1.00× | 24.08× |

### Matvec (1 kernels)

Average SeGuRu/CUDA ratio: **1.42×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| matvec_forward | small | 15.78 | 8.70 | 33.79 | 1.81× | 0.47× |
| matvec_forward | large | 808.26 | 786.43 | 63.49 | 1.03× | 12.73× |

### Norm (5 kernels)

Average SeGuRu/CUDA ratio: **1.93×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| rms_norm_forward | small | 11.68 | 6.14 | 72.70 | 1.90× | 0.16× |
| l1_norm_forward | small | 11.87 | 6.14 | 53.25 | 1.93× | 0.22× |
| l2_norm_forward | small | 11.66 | 6.14 | 68.61 | 1.90× | 0.17× |
| layer_norm_forward | small | 12.86 | 7.17 | 33.79 | 1.79× | 0.38× |
| rms_norm_forward | large | 53.52 | 40.96 | 84.99 | 1.31× | 0.63× |
| l1_norm_forward | large | 52.60 | 40.45 | 80.90 | 1.30× | 0.65× |
| l2_norm_forward | large | 52.58 | 41.98 | 79.87 | 1.25× | 0.66× |
| layer_norm_forward | large | 58.26 | 33.79 | 37.89 | 1.72× | 1.54× |
| frobenius_norm_forward | small | 14.53 | 6.14 | 32.77 | 2.37× | 0.44× |
| frobenius_norm_forward | large | 751.17 | 197.63 | 36.86 | 3.80× | 20.38× |

### Reduction (4 kernels)

Average SeGuRu/CUDA ratio: **1.99×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| sum_reduce | small | 11.73 | 6.14 | 26.62 | 1.91× | 0.44× |
| mean_reduce | small | 11.76 | 6.14 | 26.62 | 1.92× | 0.44× |
| max_reduce | small | 11.66 | 6.14 | 30.72 | 1.90× | 0.38× |
| min_reduce | small | 11.99 | 6.14 | 30.72 | 1.95× | 0.39× |
| sum_reduce | large | 20.41 | 10.24 | 27.65 | 1.99× | 0.74× |
| mean_reduce | large | 21.53 | 10.24 | 26.62 | 2.10× | 0.81× |
| max_reduce | large | 20.91 | 10.24 | 37.89 | 2.04× | 0.55× |
| min_reduce | large | 21.25 | 10.24 | 37.89 | 2.08× | 0.56× |

### Scalar (1 kernels)

Average SeGuRu/CUDA ratio: **1.89×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| scalar_multiply | small | 10.51 | 5.12 | 23.55 | 2.05× | 0.45× |
| scalar_multiply | large | 14.16 | 8.19 | 22.53 | 1.73× | 0.63× |

### Softmax (2 kernels)

Average SeGuRu/CUDA ratio: **1.87×**

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | PyTorch (µs) | SeGuRu/CUDA | SeGuRu/PyTorch |
|---|---|---|---|---|---|
| softmax_forward | small | 13.47 | 7.17 | 22.53 | 1.88× | 0.60× |
| log_softmax_forward | small | 13.45 | 6.14 | 22.53 | 2.19× | 0.60× |
| softmax_forward | large | 94.39 | 55.30 | 37.89 | 1.71× | 2.49× |
| log_softmax_forward | large | 92.44 | 53.76 | 34.82 | 1.72× | 2.65× |

## Analysis

**Most competitive categories** (lowest overhead):
- matvec: 1.42×
- cumulative: 1.49×
- elementwise: 1.87×

**Highest overhead categories:**
- argreduce: 2.34×
- loss: 2.52×
- batched_matmul: 3.01×

**Size scaling:** Overhead is similar across input sizes.
