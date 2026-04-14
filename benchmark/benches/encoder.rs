mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};
use gpu::Float4;

struct EncoderFwd<'a> {
    config: Config,
    out: gpu_host::TensorViewMut<'a, [Float4]>,
    inp: gpu_host::TensorViewMut<'a, [i32]>,
    wte: gpu_host::TensorViewMut<'a, [Float4]>,
    wpe: gpu_host::TensorViewMut<'a, [Float4]>,
}

struct EncoderBwd<'a> {
    config: Config,
    out: gpu_host::TensorViewMut<'a, [f32]>,
    inp: gpu_host::TensorViewMut<'a, [i32]>,
    wte: gpu_host::TensorViewMut<'a, [f32]>,
    wpe: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a> KernelRunner<'a> for EncoderFwd<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.seq_len * config.channel;
        let inlen = len;
        if len > u32::MAX as usize {
            return None;
        }
        let out =
            ctx.new_tensor_view(rand_float4_vec(len).as_slice()).expect("tensor alloc failed");
        let inp = ctx
            .new_tensor_view(rand_i32_in_vocab_vec(inlen, config.padded_vocab_size).as_slice())
            .expect("tensor alloc failed");
        let wte = ctx
            .new_tensor_view(rand_float4_vec(config.padded_vocab_size * config.channel).as_slice())
            .expect("tensor alloc failed");
        let wpe = ctx
            .new_tensor_view(rand_float4_vec(config.seq_len * config.channel).as_slice())
            .expect("tensor alloc failed");
        Some(Self { config, out, inp, wte, wpe })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BDIM: u32 = 512;
        let grid = (self.config.batch_size * self.config.seq_len * self.config.channel / 4)
            .div_ceil(BDIM as usize) as u32;
        gpu_host::gpu_config!(grid, 1, 1, @const BDIM, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::encoder_forward_kernel3::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.out,
            &self.inp,
            &self.wte,
            &self.wpe,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.channel as _,
        )
        .expect("kernel launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::encoder_forward_host(
                self.out.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.wte.as_devptr() as _,
                self.wpe.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
            );
        }
    }
}

impl<'a> KernelRunner<'a> for EncoderBwd<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.seq_len * config.channel;
        if len > u32::MAX as usize {
            return None;
        }
        let inlen = config.batch_size * config.seq_len;
        let out = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).expect("tensor alloc failed");
        let inp = ctx
            .new_tensor_view(rand_i32_in_vocab_vec(inlen, config.padded_vocab_size).as_slice())
            .expect("tensor alloc failed");
        let wte = ctx
            .new_tensor_view(rand_f32_vec(config.padded_vocab_size * config.channel).as_slice())
            .expect("tensor alloc failed");
        let wpe = ctx
            .new_tensor_view(rand_f32_vec(config.seq_len * config.channel).as_slice())
            .expect("tensor alloc failed");
        Some(Self { config, out, inp, wte, wpe })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BDIM: u32 = 256;
        let grid = (self.config.batch_size * self.config.seq_len * self.config.channel)
            .div_ceil(BDIM as usize) as u32;
        gpu_host::gpu_config!(grid, 1, 1, @const BDIM, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::encoder_backward_kernel::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.wte,
            &mut self.wpe,
            &self.out,
            &self.inp,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.channel as _,
        )
        .expect("kernel launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::encoder_backward_host(
                self.wte.as_devptr() as _,
                self.wpe.as_devptr() as _,
                self.out.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.channel as _,
            );
        }
    }
}

gen_bench!(EncoderFwd, "encoder_fwd", EncoderBwd, "encoder_bwd");
