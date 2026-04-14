/*
use gpu_host::cuda_ctx;
mod common;
use crate::common::random_f32_vec;

#[allow(clippy::erasing_op)]
pub fn permute_cpu(
    inp: &[f32],
    b_len: usize,
    n_len: usize,
    nh: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Allocate output tensors: shape [B, NH, N, d] flattened
    let mut q = vec![0f32; b_len * nh * n_len * d];
    let mut k = vec![0f32; b_len * nh * n_len * d];
    let mut v = vec![0f32; b_len * nh * n_len * d];

    for idx in 0..(b_len * nh * n_len * d) {
        let b = idx / (nh * n_len * d);
        let rest1 = idx % (nh * n_len * d);
        let nh_ = rest1 / (n_len * d);
        let rest2 = rest1 % (n_len * d);
        let n = rest2 / d;
        let d_ = rest2 % d;

        // Compute the index in inp: inp[b][n][0/1/2][nh_][d_]
        let inp_idx = b * n_len * 3 * nh * d + n * 3 * nh * d + 0 * nh * d + nh_ * d + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + nh * d];
        v[idx] = inp[inp_idx + 2 * nh * d];
    }

    (q, k, v)
}

#[allow(clippy::erasing_op)]
pub fn permute_backward_cpu(
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    b_len: usize,
    n_len: usize,
    nh: usize,
    d: usize,
) -> Vec<f32> {
    // Allocate output tensor: shape [B, N, 3, NH, d] flattened
    let mut dinp = vec![0f32; b_len * n_len * 3 * nh * d];

    for idx in 0..(b_len * nh * n_len * d) {
        let b = idx / (nh * n_len * d);
        let rest1 = idx % (nh * n_len * d);
        let nh_ = rest1 / (n_len * d);
        let rest2 = rest1 % (n_len * d);
        let n = rest2 / d;
        let d_ = rest2 % d;

        let inp_idx = b * n_len * 3 * nh * d + n * 3 * nh * d + 0 * nh * d + nh_ * d + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + nh * d] = dk[idx];
        dinp[inp_idx + 2 * nh * d] = dv[idx];
    }

    dinp
}

#[allow(clippy::needless_range_loop)]
pub fn unpermute_cpu(inp: &[f32], b_len: usize, n_len: usize, nh: usize, d: usize) -> Vec<f32> {
    // Allocate output tensor: shape [B, N, NH, d] flattened
    let mut out = vec![0f32; b_len * n_len * nh * d];

    for idx in 0..(b_len * nh * n_len * d) {
        let b = idx / (nh * n_len * d);
        let rest1 = idx % (nh * n_len * d);
        let nh_ = rest1 / (n_len * d);
        let rest2 = rest1 % (n_len * d);
        let n = rest2 / d;
        let d_ = rest2 % d;

        // Compute the flattened index for output: out[b][n][nh_][d_]
        let other_idx = b * n_len * nh * d + n * nh * d + nh_ * d + d_;
        out[other_idx] = inp[idx]; // __ldcs is just a load on CPU
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn test_permute_kernel(h_dinp: &[f32], b_len: u32, n_len: u32, nh_len: u32, d_len: u32) {
    let len = (b_len * nh_len * n_len * d_len) as usize;
    let mut h_dq: Vec<f32> = random_f32_vec(len);
    let mut h_dk: Vec<f32> = random_f32_vec(len);
    let mut h_dv: Vec<f32> = random_f32_vec(len);
    assert_eq!(h_dinp.len() as u32, b_len * nh_len * n_len * d_len * 3);
    cuda_ctx(0, |ctx, m| {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = b_len * nh_len * n_len * d_len;
        let num_blocks = total_threads.div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        // define GPU mem
        let mut inp = ctx.new_tensor_view(h_dinp).unwrap();
        let mut dq = ctx.new_tensor_view(h_dq.as_slice()).unwrap();
        let mut dk = ctx.new_tensor_view(h_dk.as_slice()).unwrap();
        let mut dv = ctx.new_tensor_view(h_dv.as_slice()).unwrap();
        llm_rs_gpu::permute_kernel::launch(
            config, ctx, m, &mut dq, &mut dk, &mut dv, &inp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run permute_kernel");
        let (expected_dq, expected_dk, expected_dv) = permute_cpu(
            h_dinp,
            b_len as usize,
            n_len as usize,
            nh_len as usize,
            d_len as usize,
        );
        dq.copy_to_host(&mut h_dq).unwrap();
        dk.copy_to_host(&mut h_dk).unwrap();
        dv.copy_to_host(&mut h_dv).unwrap();
        assert!(
            common::f32_eq(&h_dq, &expected_dq, 1e-8),
            "h_dq not match:\n\n{:?}\n\n{:?}",
            &h_dq[0..32],
            &expected_dq[0..32],
        );
        assert!(
            common::f32_eq(&h_dk, &expected_dk, 1e-8),
            "h_dk not match:\n\n{:?}\n\n{:?}",
            &h_dk[0..32],
            &expected_dk[0..32],
        );
        assert!(
            common::f32_eq(&h_dv, &expected_dv, 1e-8),
            "h_dv not match:\n\n{:?}\n\n{:?}",
            &h_dv[0..32],
            &expected_dv[0..32],
        );

        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::permute_kernel_backward::launch(
            config, ctx, m, &mut inp, &dq, &dk, &dv, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run permute_kernel_backward");
        let mut h_dinp2 = vec![0.0f32; h_dinp.len()];
        inp.copy_to_host(&mut h_dinp2).expect("copy to host failed");
        assert!(
            common::f32_eq(h_dinp, &h_dinp2, 1e-5),
            "h_dinp not match:\n\n{:?}\n\n{:?}",
            &h_dinp[0..32],
            &h_dinp2[0..32],
        );
    });
}

#[test]
fn test_basic() {
    const B: u32 = 2;
    const N: u32 = 4;
    const NH: u32 = 2;
    const D: u32 = 8;
    const LEN: usize = (B * N * NH * D) as usize;

    let mut h_dinp = [0.0f32; LEN * 3];
    for (i, input) in h_dinp.iter_mut().enumerate() {
        *input = i as f32;
    }
    test_permute_kernel(&h_dinp, B, N, NH, D);
}

#[test]
fn test_basic_random() {
    const B: u32 = 2;
    const N: u32 = 4;
    const NH: u32 = 2;
    const D: u32 = 8;
    const LEN: usize = (B * N * NH * D) as usize;

    let h_dinp = random_f32_vec(LEN * 3);
    test_permute_kernel(&h_dinp, B, N, NH, D);
}

#[allow(clippy::too_many_arguments)]
fn test_unpermute_kernel(h_dinp: &[f32], b_len: u32, n_len: u32, nh_len: u32, d_len: u32) {
    let len = (b_len * n_len * nh_len * d_len) as usize;
    assert_eq!(h_dinp.len(), len);
    let mut h_doutp = vec![0.0f32; len];
    cuda_ctx(0, |ctx, m| {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = b_len * nh_len * n_len * d_len;
        let num_blocks = total_threads.div_ceil(BLOCK_SIZE);
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        // define GPU mem
        let mut inp = ctx.new_tensor_view(h_dinp).unwrap();
        let mut outp = ctx.new_tensor_view(h_doutp.as_slice()).unwrap();
        llm_rs_gpu::unpermute_kernel::launch(
            config, ctx, m, &inp, &mut outp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run unpermute_kernel");
        let expected = unpermute_cpu(
            h_dinp,
            b_len as usize,
            n_len as usize,
            nh_len as usize,
            d_len as usize,
        );
        outp.copy_to_host(&mut h_doutp).unwrap();
        assert!(
            common::f32_eq(&h_doutp, &expected, 1e-8),
            "h_doutp not match:\n\n{:?}\n\n{:?}",
            &h_doutp[0..32],
            &expected[0..32],
        );

        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BLOCK_SIZE, 1, 1, 0);
        llm_rs_gpu::unpermute_kernel_backward::launch(
            config, ctx, m, &mut inp, &outp, b_len, n_len, nh_len, d_len,
        )
        .expect("Failed to run unpermute_kernel_backward");
        let mut h_dinp2 = vec![0.0f32; h_dinp.len()];
        inp.copy_to_host(&mut h_dinp2).expect("copy to host failed");
        assert!(
            common::f32_eq(h_dinp, &h_dinp2, 1e-5),
            "h_dinp not match:\n\n{:?}\n\n{:?}",
            &h_dinp[0..32],
            &h_dinp2[0..32],
        );
    });
}

#[test]
fn test_basic_unpermute() {
    const B: u32 = 4;
    const N: u32 = 8;
    const NH: u32 = 2;
    const D: u32 = 16;
    const LEN: usize = (B * N * NH * D) as usize;

    let h_inp = random_f32_vec(LEN);
    test_unpermute_kernel(&h_inp, B, N, NH, D);
}
*/

mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

struct Permute<'a, const FORWARD: bool> {
    config: Config,
    q: gpu_host::TensorViewMut<'a, [f32]>,
    k: gpu_host::TensorViewMut<'a, [f32]>,
    v: gpu_host::TensorViewMut<'a, [f32]>,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a, const FORWARD: bool> Permute<'a, FORWARD> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let qkv_len = config.batch_size * config.seq_len * config.channel;
        let inp_len = qkv_len * 3;
        let inp =
            ctx.new_tensor_view(rand_f32_vec(inp_len).as_slice()).expect("tensor alloc failed");
        let q = ctx.new_tensor_view(rand_f32_vec(qkv_len).as_slice()).expect("tensor alloc failed");
        let k = ctx.new_tensor_view(rand_f32_vec(qkv_len).as_slice()).expect("tensor alloc failed");
        let v = ctx.new_tensor_view(rand_f32_vec(qkv_len).as_slice()).expect("tensor alloc failed");
        Some(Self { config, q, k, v, inp })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = self.config.batch_size
            * self.config.num_heads
            * self.config.seq_len
            * (self.config.channel / self.config.num_heads);
        let num_blocks = total_threads.div_ceil(BLOCK_SIZE as usize);
        gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BLOCK_SIZE, 1, 1, 0)
    }
}
impl<'a> KernelRunner<'a> for Permute<'a, true> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        Self::new(ctx, config)
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        self.launch_config()
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::permute_kernel::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &self.inp,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.num_heads as _,
            self.config.head_size as _,
        )
        .expect("Failed to run permute_kernel");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::permute_kernel_host(
                self.q.as_devptr() as _,
                self.k.as_devptr() as _,
                self.v.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.num_heads as _,
                self.config.head_size as _,
            );
        }
    }
}

impl<'a> KernelRunner<'a> for Permute<'a, false> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        Self::new(ctx, config)
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        self.launch_config()
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::permute_kernel_backward::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.inp,
            &self.q,
            &self.k,
            &self.v,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.num_heads as _,
            self.config.head_size as _,
        )
        .expect("Failed to run permute_kernel_backward");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::permute_kernel_backward_host(
                self.inp.as_devptr() as _,
                self.q.as_devptr() as _,
                self.k.as_devptr() as _,
                self.v.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.num_heads as _,
                self.config.head_size as _,
            );
        }
    }
}

type PermuteForward<'a> = Permute<'a, true>;
type PermuteBackward<'a> = Permute<'a, false>;

struct Unpermute<'a, const FORWARD: bool> {
    config: Config,
    outp: gpu_host::TensorViewMut<'a, [f32]>,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a, const FORWARD: bool> Unpermute<'a, FORWARD> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.seq_len * config.channel;
        let inp = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let outp = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        Some(Self { config, outp, inp })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BLOCK_SIZE: u32 = 256;
        let total_threads = self.config.batch_size * self.config.seq_len * self.config.channel;
        let num_blocks = total_threads.div_ceil(BLOCK_SIZE as usize);
        gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BLOCK_SIZE, 1, 1, 0)
    }
}

impl<'a> KernelRunner<'a> for Unpermute<'a, true> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        Self::new(ctx, config)
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        self.launch_config()
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::unpermute_kernel::launch(
            self.launch_config(),
            ctx,
            m,
            &self.inp,
            &mut self.outp,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.num_heads as _,
            self.config.head_size as _,
        )
        .expect("Failed to run unpermute_kernel");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::unpermute_kernel_host(
                self.inp.as_devptr() as _,
                self.outp.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.num_heads as _,
                self.config.head_size as _,
            );
        }
    }
}

impl<'a> KernelRunner<'a> for Unpermute<'a, false> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        Self::new(ctx, config)
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        self.launch_config()
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::unpermute_kernel_backward::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.inp,
            &self.outp,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.num_heads as _,
            self.config.head_size as _,
        )
        .expect("Failed to run unpermute_kernel_backward");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::unpermute_kernel_backward_host(
                self.inp.as_devptr() as _,
                self.outp.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.num_heads as _,
                self.config.head_size as _,
            );
        }
    }
}

type UnpermuteForward<'a> = Unpermute<'a, true>;
type UnpermuteBackward<'a> = Unpermute<'a, false>;

gen_bench!(
    PermuteForward,
    "permute-fwd",
    PermuteBackward,
    "permute-bwd",
    UnpermuteForward,
    "unpermute-fwd",
    UnpermuteBackward,
    "unpermute-bwd"
);
