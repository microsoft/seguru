mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};
use gpu::prelude::*;

struct SoftMaxForward<'a> {
    config: Config,
    att: gpu_host::TensorViewMut<'a, [Float4]>,
    preatt: gpu_host::TensorViewMut<'a, [f32]>,
    scale: f32,
}

impl<'a> KernelRunner<'a> for SoftMaxForward<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.num_heads * config.seq_len * config.seq_len;
        if len > u32::MAX as usize {
            return None;
        }
        let att =
            ctx.new_tensor_view(rand_float4_vec(len / 4).as_slice()).expect("tensor alloc failed");
        let preatt =
            ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let scale = 1.0f32 / (config.head_size as f32).sqrt();
        Some(Self { config, att, preatt, scale })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BSIZE: u32 = 256;
        let grid_size = (self.config.batch_size * self.config.num_heads * self.config.seq_len * 32)
            .div_ceil(BSIZE as usize) as u32;
        gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::softmax_forward_kernel5::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.preatt,
            self.scale,
            &self.att,
            (self.config.batch_size * self.config.num_heads) as _,
            self.config.seq_len as _,
        )
        .expect("failed to launch softmax_forward_kernel5");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::softmax_forward_host(
                self.preatt.as_devptr() as _,
                self.att.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.num_heads as _,
                self.scale,
            );
        }
    }
}

// softmax_autoregressive_backward_kernel
struct SoftMaxBack<'a> {
    config: Config,
    dpreatt: gpu_host::TensorViewMut<'a, [f32]>,
    datt: gpu_host::TensorViewMut<'a, [f32]>,
    att: gpu_host::TensorViewMut<'a, [f32]>,
    scale: f32,
}

impl<'a> KernelRunner<'a> for SoftMaxBack<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.num_heads * config.seq_len * config.seq_len;
        if len > u32::MAX as usize {
            return None;
        }
        let dpreatt =
            ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let datt = ctx.new_tensor_view(vec![0f32; len].as_slice()).expect("tensor alloc failed");
        let att = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let scale = 1.0f32 / (config.head_size as f32).sqrt();
        Some(Self { config, dpreatt, datt, att, scale })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        gpu_host::gpu_config!(
            (self.config.seq_len / 4) as u32,
            (self.config.batch_size) as u32,
            1,
            256,
            1,
            1,
            0
        )
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::softmax_autoregressive_backward_kernel::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.dpreatt,
            &self.datt,
            &self.att,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.channel as _,
            self.scale,
        )
        .expect("failed to launch softmax_autoregressive_backward_kernel");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::softmax_autoregressive_backward_host(
                self.dpreatt.as_devptr() as _,
                self.datt.as_devptr() as _,
                self.att.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
                self.scale,
            );
        }
    }
}

gen_bench!(SoftMaxForward, "softmax-fwd", SoftMaxBack, "softmax-bwd");
