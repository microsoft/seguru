//! HEonGPU primitives on SeGuRu: modular arithmetic, negacyclic NTT and
//! element-wise ciphertext operations over Z_q for a word-size prime q.

pub mod arith;
pub mod cpu;
pub mod modular;
pub mod ntt;

#[cfg(feature = "bench")]
pub mod cuda_ffi;

#[cfg(test)]
mod tests;
