mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

/*cuda_ctx(0, |ctx, m| {
        let mut d_params = ctx
            .new_tensor_view(params.as_slice())
            .expect("alloc failed");
        let d_grads = ctx.new_tensor_view(grads.as_slice()).expect("alloc failed");
        let mut d_m = ctx
            .new_tensor_view(m_memory.as_slice())
            .expect("alloc failed");
        let mut d_v = ctx
            .new_tensor_view(v_memory.as_slice())
            .expect("alloc failed");
        let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
        llm_rs_gpu::adamw_kernel2::launch(
            config,
            ctx,
            m,
            &mut d_params,
            &d_grads,
            &mut d_m,
            &mut d_v,
            num_params as _,
            learning_rate,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            weight_decay,
        )
        .expect("Failed to run adamw_kernel2");
        d_params
            .copy_to_host(&mut new_params_gpu)
            .expect("copy to host failed");
        d_m.copy_to_host(&mut new_m_gpu)
            .expect("copy to host failed");
        d_v.copy_to_host(&mut new_v_gpu)
            .expect("copy to host failed");
    });
*/
struct AdamWKernel<'a> {
    config: Config,
    params: gpu_host::TensorViewMut<'a, [f32]>,
    grads: gpu_host::TensorViewMut<'a, [f32]>,
    m_mem: gpu_host::TensorViewMut<'a, [f32]>,
    v_mem: gpu_host::TensorViewMut<'a, [f32]>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    eps: f32,
    weight_decay: f32,
}

impl<'a> KernelRunner<'a> for AdamWKernel<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let num_params = config.get_params_sizes().iter().sum();
        if num_params > u32::MAX as usize {
            return None;
        }
        let params =
            ctx.new_tensor_view(rand_f32_vec(num_params).as_slice()).expect("tensor alloc failed");
        let grads =
            ctx.new_tensor_view(rand_f32_vec(num_params).as_slice()).expect("tensor alloc failed");
        let m_mem =
            ctx.new_tensor_view(rand_f32_vec(num_params).as_slice()).expect("tensor alloc failed");
        let v_mem =
            ctx.new_tensor_view(rand_f32_vec(num_params).as_slice()).expect("tensor alloc failed");
        let learning_rate = 3e-4;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let step = 12;
        let beta1_correction = 1.0 - beta1.powi(step);
        let beta2_correction = 1.0 - beta2.powi(step);
        let eps = 1e-8;
        let weight_decay = 1e-2;
        Some(Self {
            config,
            params,
            grads,
            m_mem,
            v_mem,
            learning_rate,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            weight_decay,
        })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BSIZE: u32 = 512;
        let num_params = self.config.get_params_sizes().iter().sum::<usize>();
        let grid_size = (num_params as u32 + BSIZE - 1) / BSIZE;
        gpu_host::gpu_config!(grid_size, 1, 1, @const BSIZE, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        let num_params = self.config.get_params_sizes().iter().sum::<usize>();
        llm_rs_gpu::adamw_kernel2::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.params,
            &self.grads,
            &mut self.m_mem,
            &mut self.v_mem,
            num_params as _,
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.beta1_correction,
            self.beta2_correction,
            self.eps,
            self.weight_decay,
        )
        .expect("Failed to run adamw_kernel2");
    }

    fn c_fn(&mut self) {
        let num_params = self.config.get_params_sizes().iter().sum::<usize>();
        unsafe {
            llmc::adamw_kernel2_host(
                self.params.as_devptr() as _,
                self.grads.as_devptr() as _,
                self.m_mem.as_devptr() as _,
                self.v_mem.as_devptr() as _,
                num_params as _,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.beta1_correction,
                self.beta2_correction,
                self.eps,
                self.weight_decay,
            );
        }
    }
}

gen_bench!(AdamWKernel, "adamw");
