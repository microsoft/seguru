void matmul_backward_bias_kernel4_host(float *dbias, const float *dout, int B,
                                       int T, int OC);
void matmul_forward_host(float *out, const float *inp, const float *weight,
                         const float *bias, int B, int T, int C, int OC);
void gelu_backward(float *dinp, const float *inp, const float *dout,
                   const int N);
void gelu_forward(float *out, const float *inp, int N);

void softmax_forward_host(float *out, const float *inp, int B, int T, int NH,
                          float scale);
void softmax_autoregressive_backward_host(float *dpreatt, const float *datt,
                                          const float *att, int B, int T, int C,
                                          float scale);
void layernorm_backward_host(float *dinp, float *dweight, float *dbias,
                             const float *dout, const float *inp,
                             const float *weight, const float *mean,
                             const float *rstd, int B, int T, int C);
void layernorm_forward_host(float *out, float *mean, float *rstd, float *inp,
                            float *weight, float *bias, int B, int T, int C);
void fused_classifier_host(float *logits, float *losses, const float *dlosses,
                           const int *targets, int B, int T, int V, int P);
void permute_kernel_host(float *q, float *k, float *v, float *inp, int B, int T,
                         int NH, int HS);
void permute_kernel_backward_host(float *dinp, float *dq, float *dk, float *dv,
                                  int B, int T, int NH, int HS);
void empty_host(int gdim_x, int gdim_y, int gdim_z, int bdim_x, int bdim_y, int bdim_z, int shared_size);
void unpermute_kernel_backward_host(float *dinp, float *dout, int B, int T,
                                    int NH, int HS);
void unpermute_kernel_host(float *out, float *inp, int B, int T, int NH,
                           int HS);
void adamw_kernel2_host(float *params, float *grads, float *m, float *v,
                        int num_parameters, float learning_rate, float beta1,
                        float beta2, float beta1_correction,
                        float beta2_correction, float eps, float weight_decay);
void encoder_forward_host(float *out, const int *inp, const float *wte,
                          const float *wpe, int B, int T, int C);
void encoder_backward_host(float *dwte, float *dwpe, const float *dout,
                           const int *inp, int B, int T, int C);
void residual_forward_host(float *out, float *inp1, float *inp2, int N);