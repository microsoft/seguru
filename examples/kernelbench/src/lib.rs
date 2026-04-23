#![no_std]
#![allow(clippy::too_many_arguments)]
#![deny(clippy::cast_possible_truncation)]

pub mod elementwise;
pub mod gelu_variants;
pub mod softmax;
pub mod matmul;
pub mod matvec;
pub mod reduction;
pub mod argreduce;
pub mod norm;
pub mod loss;
pub mod cumulative;
