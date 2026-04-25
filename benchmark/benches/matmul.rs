mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};
use gpu::Float4;

struct MatMulBack<'a> {
    bias: gpu_host::TensorViewMut<'a, [f32]>,
    dout: gpu_host::TensorViewMut<'a, [f32]>,
    config: Config,
}

impl<'a> KernelRunner<'a> for MatMulBack<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let bias = ctx
            .new_tensor_view(rand_f32_vec(config.out_channel).as_slice())
            .expect("tensor alloc failed");
        let dout = ctx
            .new_tensor_view(
                rand_f32_vec(config.batch_size * config.seq_len * config.out_channel).as_slice(),
            )
            .expect("tensor alloc failed");
        Some(Self { bias, dout, config })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BLOCK_SIZE: u32 = 1024;
        let out_channel = self.config.out_channel;
        const SMEM_SIZE: u32 = BLOCK_SIZE * std::mem::size_of::<f32>() as u32;
        let grid_size = (out_channel as u32).div_ceil(32);
        gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BLOCK_SIZE, 1, 1, @const SMEM_SIZE)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::matmul_backward_bias_kernel4::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.bias,
            &self.dout,
            self.config.batch_size as _, // batch size
            self.config.seq_len as _,    // seq length
            self.config.out_channel as _,
        )
        .expect("launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::matmul_backward_bias_kernel4_host(
                self.bias.as_devptr() as _,
                self.dout.as_devptr() as _,
                self.config.batch_size as _, // batch size
                self.config.seq_len as _,    // seq length
                self.config.out_channel as _,
            );
        }
    }
}

struct MatMulForward<'a> {
    bias: gpu_host::TensorViewMut<'a, [Float4]>,
    out: gpu_host::TensorViewMut<'a, [Float4]>,
    inp: gpu_host::TensorViewMut<'a, [Float4]>,
    weight: gpu_host::TensorViewMut<'a, [Float4]>,
    config: Config,
}

impl<'a> KernelRunner<'a> for MatMulForward<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let channel = config.channel;
        let bias =
            ctx.new_tensor_view(rand_float4_vec(channel).as_slice()).expect("tensor alloc failed");
        let out = ctx
            .new_tensor_view(
                rand_float4_vec(config.batch_size * config.seq_len * config.out_channel).as_slice(),
            )
            .expect("tensor alloc failed");
        let inp = ctx
            .new_tensor_view(
                rand_float4_vec(config.batch_size * config.seq_len * channel).as_slice(),
            )
            .expect("tensor alloc failed");
        let weight = ctx
            .new_tensor_view(rand_float4_vec(channel * channel).as_slice())
            .expect("tensor alloc failed");
        Some(Self { bias, out, inp, weight, config })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const SQRT_BLOCK_SIZE: u32 = 16;
        let n = self.config.batch_size * self.config.seq_len;
        let out_channel = self.config.out_channel;
        let grid_x = (n as u32).div_ceil(8 * SQRT_BLOCK_SIZE);
        let grid_y = (out_channel as u32).div_ceil(8 * SQRT_BLOCK_SIZE);
        gpu_host::gpu_config!(grid_x as u32, grid_y as u32, 1, @const SQRT_BLOCK_SIZE, @const SQRT_BLOCK_SIZE, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::matmul_forward_kernel4::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.out,
            &self.inp,
            &self.weight,
            &self.bias,
            self.config.channel as _,
            self.config.out_channel as _,
        )
        .expect("launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::matmul_forward_host(
                self.out.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.weight.as_devptr() as _,
                self.bias.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
                self.config.out_channel as _,
            );
        }
    }
}

gen_bench!(MatMulBack, "matmul-bwd", MatMulForward, "matmul-fwd");
