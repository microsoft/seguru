//! SeGuRu ports translated from the raw-CUDA arm (not from the PyTorch
//! reference). Each submodule mirrors the corresponding `cuda/<name>.cu`
//! kernel as closely as the skill doc allows.
//!
//! Used to isolate the "translation" axis from the "from-scratch design"
//! axis when evaluating LLM-as-porter for SeGuRu.

pub mod argmax_dim;
pub mod avg_pool1d;
pub mod cumsum;
pub mod gelu;
pub mod l1_norm;
pub mod l2_norm;
pub mod layer_norm;
pub mod leaky_relu;
pub mod log_softmax;
pub mod max_dim;
pub mod max_pool1d;
pub mod mean_dim;
pub mod min_dim;
pub mod mse_loss;
pub mod relu;
pub mod rms_norm;
pub mod sigmoid;
pub mod softmax;
pub mod softplus;
pub mod sum_dim;
pub mod swish;
pub mod tanh;
