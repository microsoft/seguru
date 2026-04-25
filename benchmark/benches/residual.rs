mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

struct ResidualForward<'a> {
    out: gpu_host::TensorViewMut<'a, [f32]>,
    inp1: gpu_host::TensorViewMut<'a, [f32]>,
    inp2: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a> KernelRunner<'a> for ResidualForward<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.batch_size * config.seq_len * config.channel;
        let out = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).unwrap();
        let inp1 = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).unwrap();
        let inp2 = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).unwrap();
        Some(Self { out, inp1, inp2 })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BSIZE: u32 = 256;
        let grid_size = self.inp1.len().div_ceil(BSIZE as usize) as u32;
        gpu_host::gpu_config!(grid_size, 1, 1, @const BSIZE as u32, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::residual_forward_kernel::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.out,
            &self.inp1,
            &self.inp2,
            self.inp1.len() as u32,
        )
        .expect("launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::residual_forward_host(
                self.out.as_devptr() as _,
                self.inp1.as_devptr() as _,
                self.inp2.as_devptr() as _,
                self.inp1.len() as _,
            );
        }
    }
}

gen_bench!(ResidualForward, "residual-fwd");
