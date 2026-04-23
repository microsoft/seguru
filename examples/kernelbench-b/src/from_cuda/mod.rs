//! SeGuRu ports translated from the raw-CUDA arm (not from the PyTorch
//! reference). Each submodule mirrors the corresponding `cuda/<name>.cu`
//! kernel as closely as the skill doc allows.
//!
//! Used to isolate the "translation" axis from the "from-scratch design"
//! axis when evaluating LLM-as-porter for SeGuRu.

pub mod leaky_relu;
pub mod tanh;
pub mod rms_norm;
pub mod relu;
pub mod sigmoid;
pub mod gelu;
pub mod softmax;
pub mod layer_norm;
pub mod sum_dim;
pub mod l2_norm;
