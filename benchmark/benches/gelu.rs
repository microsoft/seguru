mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

struct GeLu<'a, const FORWARD: bool> {
    out: gpu_host::TensorViewMut<'a, [f32]>,
    inp: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a, const FORWARD: bool> GeLu<'a, FORWARD> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let len = config.num_layers * config.batch_size * config.seq_len * config.channel * 4;
        let out = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).unwrap();
        let inp = ctx.new_tensor_view(rand_f32_vec(len).as_slice()).unwrap();
        Some(Self { out, inp })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BSIZE: u32 = 128;
        let grid_size = self.inp.len().div_ceil(BSIZE as usize) as u32;
        gpu_host::gpu_config!(grid_size, 1, 1, @const BSIZE as u32, 1, 1, 0)
    }
}

impl<'a> KernelRunner<'a> for GeLu<'a, true> {
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
        llm_rs_gpu::gelu_forward_kernel::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.out,
            &self.inp,
            self.inp.len() as u32,
        )
        .expect("launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::gelu_forward(
                self.out.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.inp.len() as _,
            );
        }
    }
}

impl<'a> KernelRunner<'a> for GeLu<'a, false> {
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
        llm_rs_gpu::gelu_backward_kernel::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.inp,
            &self.out,
            self.out.len() as u32,
        )
        .expect("launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::gelu_backward(
                self.inp.as_devptr() as _,
                self.inp.as_devptr() as _,
                self.out.as_devptr() as _,
                self.inp.len() as _,
            );
        }
    }
}

type GeLuForward<'a> = GeLu<'a, true>;
type GeLuBackward<'a> = GeLu<'a, false>;
gen_bench!(GeLuForward, "gelu-fwd", GeLuBackward, "gelu-bwd");
