mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

struct LayerNormForward<'a> {
    config: Config,
    out: gpu_host::TensorViewMut<'a, [f32]>,
    mean: gpu_host::TensorViewMut<'a, [f32]>,
    rstd: gpu_host::TensorViewMut<'a, [f32]>,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
    weight: gpu_host::TensorViewMut<'a, [f32]>,
    bias: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a> KernelRunner<'a> for LayerNormForward<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.seq_len * config.channel;
        if len > u32::MAX as usize {
            return None;
        }
        let out = ctx.new_tensor_view(vec![0f32; len].as_slice()).expect("tensor alloc failed");
        let mean = ctx
            .new_tensor_view(vec![0f32; config.batch_size * config.seq_len].as_slice())
            .expect("tensor alloc failed");
        let rstd = ctx
            .new_tensor_view(vec![0f32; config.batch_size * config.seq_len].as_slice())
            .expect("tensor alloc failed");
        let inp = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let weight = ctx
            .new_tensor_view(rand_f32_vec(config.channel).as_slice())
            .expect("tensor alloc failed");
        let bias = ctx
            .new_tensor_view(rand_f32_vec(config.channel).as_slice())
            .expect("tensor alloc failed");
        Some(Self { config, out, mean, rstd, inp, weight, bias })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BDIM: u32 = 512;
        let grid =
            (self.config.batch_size * self.config.seq_len * 32).div_ceil(BDIM as usize) as u32;
        gpu_host::gpu_config!(grid, 1, 1, @const BDIM, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::layernorm_forward_kernel3::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.out,
            &mut self.mean,
            &mut self.rstd,
            &self.inp,
            &self.weight,
            &self.bias,
            (self.config.batch_size * self.config.seq_len) as _,
            self.config.channel as _,
        )
        .expect("failed to launch layernorm_forward_kernel3");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::layernorm_forward_host(
                self.out.as_devptr() as _,
                self.mean.as_devptr() as _,
                self.rstd.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.weight.as_devptr() as _,
                self.bias.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
            );
        }
    }
}

struct LayerNormBack<'a> {
    config: Config,
    dinp: gpu_host::TensorViewMut<'a, [f32]>,
    dweight: gpu_host::TensorViewMut<'a, [f32]>,
    dbias: gpu_host::TensorViewMut<'a, [f32]>,
    dout: gpu_host::TensorViewMut<'a, [f32]>,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
    weight: gpu_host::TensorViewMut<'a, [f32]>,
    mean: gpu_host::TensorViewMut<'a, [f32]>,
    rstd: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a> KernelRunner<'a> for LayerNormBack<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.seq_len * config.channel;
        if len > u32::MAX as usize {
            return None;
        }
        let dinp = ctx.new_tensor_view(vec![0f32; len].as_slice()).expect("tensor alloc failed");
        let dweight = ctx
            .new_tensor_view(vec![0f32; config.channel].as_slice())
            .expect("tensor alloc failed");
        let dbias = ctx
            .new_tensor_view(vec![0f32; config.channel].as_slice())
            .expect("tensor alloc failed");
        let dout = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let inp = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let weight = ctx
            .new_tensor_view(rand_f32_vec(config.channel).as_slice())
            .expect("tensor alloc failed");
        let mean = ctx
            .new_tensor_view(vec![0f32; config.batch_size * config.seq_len].as_slice())
            .expect("tensor alloc failed");
        let rstd = ctx
            .new_tensor_view(vec![0f32; config.batch_size * config.seq_len].as_slice())
            .expect("tensor alloc failed");
        Some(Self { config, dinp, dweight, dbias, dout, inp, weight, mean, rstd })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BDIM: u32 = 512;
        let grid =
            (self.config.batch_size * self.config.seq_len * 32).div_ceil(BDIM as usize) as u32;
        let shared_mem_size = 2 * self.config.channel * std::mem::size_of::<f32>();
        gpu_host::gpu_config!(grid, 1, 1, @const BDIM, 1, 1, shared_mem_size as u32)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::layernorm_backward_kernel2::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.dinp,
            &mut self.dweight,
            &mut self.dbias,
            &self.dout,
            &self.inp,
            &self.weight,
            &self.mean,
            &self.rstd,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.channel as _,
        )
        .expect("failed to launch layernorm_backward_kernel2");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::layernorm_backward_host(
                self.dinp.as_devptr() as _,
                self.dweight.as_devptr() as _,
                self.dbias.as_devptr() as _,
                self.dout.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.weight.as_devptr() as _,
                self.mean.as_devptr() as _,
                self.rstd.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
            );
        }
    }
}

gen_bench!(LayerNormForward, "layernorm-fwd", LayerNormBack, "layernorm-bwd");
