//! Elementwise activations.
//!
//! Every activation is pure bandwidth, so all of them share one kernel shape:
//!
//! * the buffers are typed `[Float4]`, so each access is a single 128-bit
//!   transaction instead of four 32-bit ones;
//! * each thread owns [`VEC_PER_THREAD`] `Float4`s (16 elements), strided by the
//!   grid so every warp access stays perfectly coalesced, and the unroller
//!   interleaves the four independent loads to hide latency;
//! * the host pads the buffers to a whole number of CTA tiles, so there is no
//!   tail predicate and the `reshape_map!` proves the writes disjoint.
//!
//! The transcendental helpers below deliberately call
//! [`GPUDeviceFloatIntrinsics`] through UFCS. `f32` also has inherent `exp` /
//! `tanh` / `log` methods from `std`, which method resolution would pick in
//! preference to the trait; those lower to `libm` calls that do not exist on
//! the device.

use crunchy::unroll;
use gpu::*;

pub use crate::util::{ELEMS_PER_CTA, EW_BLOCK, VEC_PER_THREAD};

/// Grid size used for `n` elements.
pub fn grid_for(n: usize) -> u32 {
    n.div_ceil(ELEMS_PER_CTA).max(1) as u32
}

/// Padded element count for `n` elements.
pub fn padded_len(n: usize) -> usize {
    grid_for(n) as usize * ELEMS_PER_CTA
}

#[gpu::device]
#[inline(always)]
pub(crate) fn dexp(x: f32) -> f32 {
    GPUDeviceFloatIntrinsics::exp(x)
}

#[gpu::device]
#[inline(always)]
pub(crate) fn dlog(x: f32) -> f32 {
    GPUDeviceFloatIntrinsics::log(x)
}

#[gpu::device]
#[inline(always)]
pub(crate) fn dtanh(x: f32) -> f32 {
    GPUDeviceFloatIntrinsics::tanh(x)
}

#[gpu::device]
#[inline(always)]
pub(crate) fn drsqrt(x: f32) -> f32 {
    GPUDeviceFloatIntrinsics::rsqrt(x)
}

macro_rules! elementwise {
    (
        $(#[$meta:meta])*
        $name:ident, $kernel:ident, $cpu:ident, $dev:ident,
        |$x:ident| $gpu_body:expr, |$xc:ident| $cpu_body:expr
    ) => {
        #[gpu::device]
        #[inline(always)]
        fn $dev($x: f32) -> f32 {
            $gpu_body
        }

        $(#[$meta])*
        #[gpu::cuda_kernel]
        pub fn $kernel(x: &[Float4], y: &mut [Float4]) {
            assert!(Config::BDIM_X == EW_BLOCK);
            let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
            let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
            let mut out =
                chunk_mut(y, reshape_map!([VEC_PER_THREAD] | [nthreads] => layout: [t0, i0]));
            unroll! {
                for k in 0..4 {
                    let v = x[(gid + (k as u32) * nthreads) as usize];
                    out[k as u32] =
                        Float4::new([$dev(v[0]), $dev(v[1]), $dev(v[2]), $dev(v[3])]);
                }
            }
        }

        /// CPU reference implementation.
        pub fn $cpu(x: &[f32]) -> Vec<f32> {
            x.iter().map(|&$xc| $cpu_body).collect()
        }

        $(#[$meta])*
        pub fn $name(x: &[f32]) -> Vec<f32> {
            let n = x.len();
            let padded = padded_len(n);
            let h4 = crate::util::to_float4_padded(x, padded);
            let grid = grid_for(n);
            gpu_host::cuda_ctx(0, |ctx, m| {
                let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
                let zeros = vec![Float4::default(); h4.len()];
                let mut d_y = ctx.new_tensor_view::<[Float4]>(&zeros).unwrap();
                let cfg = gpu_host::gpu_config!(grid, 1, 1, @const EW_BLOCK, 1, 1, 0);
                $kernel::launch(cfg, ctx, m, &d_x, &mut d_y).unwrap();
                let mut h_y = vec![Float4::default(); h4.len()];
                d_y.copy_to_host(&mut h_y).unwrap();
                crate::util::from_float4(&h_y, n)
            })
        }
    };
}

/// sqrt(2/pi), the GELU tanh-approximation scaling factor.
const GELU_C: f32 = 0.797_884_6;

elementwise!(
    /// `relu(x) = max(x, 0)`
    relu, relu_kernel, relu_cpu, relu_dev,
    |x| x.max(0.0),
    |x| x.max(0.0)
);

elementwise!(
    /// `gelu(x)`, the tanh approximation (PyTorch's `approximate="tanh"`).
    gelu, gelu_kernel, gelu_cpu, gelu_dev,
    |x| 0.5 * x * (1.0 + dtanh(GELU_C * (x + 0.044715 * x * x * x))),
    |x| 0.5 * x * (1.0 + (GELU_C * (x + 0.044715 * x * x * x)).tanh())
);

elementwise!(
    /// `sigmoid(x) = 1 / (1 + exp(-x))`
    sigmoid, sigmoid_kernel, sigmoid_cpu, sigmoid_dev,
    |x| 1.0 / (1.0 + dexp(-x)),
    |x| 1.0 / (1.0 + (-x).exp())
);

elementwise!(
    /// `tanh(x)`
    tanh, tanh_kernel, tanh_cpu, tanh_dev,
    |x| dtanh(x),
    |x| x.tanh()
);

elementwise!(
    /// `swish(x) = silu(x) = x * sigmoid(x)`
    swish, swish_kernel, swish_cpu, swish_dev,
    |x| x / (1.0 + dexp(-x)),
    |x| x / (1.0 + (-x).exp())
);

elementwise!(
    /// `softplus(x) = ln(1 + exp(x))`, evaluated in the numerically stable form
    /// `max(x, 0) + ln(1 + exp(-|x|))`.
    softplus, softplus_kernel, softplus_cpu, softplus_dev,
    |x| x.max(0.0) + dlog(1.0 + dexp(-x.max(-x))),
    |x| x.max(0.0) + (1.0 + (-(x.max(-x))).exp()).ln()
);

/// `leaky_relu(x) = x if x > 0 else slope * x`.
///
/// Written as `max(x, slope * x)`, which is branch-free and exact for
/// `0 <= slope <= 1`.
#[gpu::cuda_kernel]
pub fn leaky_relu_kernel(x: &[Float4], y: &mut [Float4], slope: f32) {
    assert!(Config::BDIM_X == EW_BLOCK);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut out = chunk_mut(y, reshape_map!([VEC_PER_THREAD] | [nthreads] => layout: [t0, i0]));
    unroll! {
        for k in 0..4 {
            let v = x[(gid + (k as u32) * nthreads) as usize];
            out[k as u32] = Float4::new([
                v[0].max(slope * v[0]),
                v[1].max(slope * v[1]),
                v[2].max(slope * v[2]),
                v[3].max(slope * v[3]),
            ]);
        }
    }
}

pub fn leaky_relu_cpu(x: &[f32], slope: f32) -> Vec<f32> {
    x.iter().map(|&v| if v > 0.0 { v } else { slope * v }).collect()
}

pub fn leaky_relu(x: &[f32], slope: f32) -> Vec<f32> {
    assert!((0.0..=1.0).contains(&slope), "the max() formulation needs 0 <= slope <= 1");
    let n = x.len();
    let padded = padded_len(n);
    let h4 = crate::util::to_float4_padded(x, padded);
    let grid = grid_for(n);
    gpu_host::cuda_ctx(0, |ctx, m| {
        let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
        let zeros = vec![Float4::default(); h4.len()];
        let mut d_y = ctx.new_tensor_view::<[Float4]>(&zeros).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const EW_BLOCK, 1, 1, 0);
        leaky_relu_kernel::launch(cfg, ctx, m, &d_x, &mut d_y, slope).unwrap();
        let mut h_y = vec![Float4::default(); h4.len()];
        d_y.copy_to_host(&mut h_y).unwrap();
        crate::util::from_float4(&h_y, n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, sample};

    fn check(gpu: impl Fn(&[f32]) -> Vec<f32>, cpu: impl Fn(&[f32]) -> Vec<f32>, name: &str) {
        // A ragged length exercises the host-side padding; an exact multiple of
        // the CTA tile exercises the fast path.
        for &n in &[1usize, 4095, ELEMS_PER_CTA, 3 * ELEMS_PER_CTA + 17, 1 << 16] {
            let x = sample(n, 7);
            assert_close(&gpu(&x), &cpu(&x), 1e-5, &format!("{name} n={n}"));
        }
    }

    #[test]
    fn relu_matches_cpu() {
        check(relu, relu_cpu, "relu");
    }

    #[test]
    fn gelu_matches_cpu() {
        check(gelu, gelu_cpu, "gelu");
    }

    #[test]
    fn sigmoid_matches_cpu() {
        check(sigmoid, sigmoid_cpu, "sigmoid");
    }

    #[test]
    fn tanh_matches_cpu() {
        check(tanh, tanh_cpu, "tanh");
    }

    #[test]
    fn swish_matches_cpu() {
        check(swish, swish_cpu, "swish");
    }

    #[test]
    fn softplus_matches_cpu() {
        check(softplus, softplus_cpu, "softplus");
    }

    #[test]
    fn leaky_relu_matches_cpu() {
        for &n in &[1usize, 4095, ELEMS_PER_CTA, 3 * ELEMS_PER_CTA + 17] {
            let x = sample(n, 11);
            assert_close(
                &leaky_relu(&x, 0.01),
                &leaky_relu_cpu(&x, 0.01),
                1e-5,
                &format!("leaky_relu n={n}"),
            );
        }
    }
}
