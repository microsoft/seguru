mod common;

use std::time::Duration;

use common::*;
use criterion::{Criterion, criterion_group, criterion_main};

struct FusedClassifier<'a> {
    config: Config,
    logits: gpu_host::TensorViewMut<'a, [f32]>,
    losses: gpu_host::TensorViewMut<'a, [f32]>,
    dlosses: gpu_host::TensorViewMut<'a, [f32]>,
    targets: gpu_host::TensorViewMut<'a, [i32]>,
    empty: gpu_host::TensorViewMut<'a, [f32]>,
}

impl<'a> KernelRunner<'a> for FusedClassifier<'a> {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        let batch_size = config.batch_size;
        let seq_len = config.seq_len;
        let padded_vocab_size = config.padded_vocab_size;
        let bt_len = batch_size * seq_len;
        let bt_padded_len: usize = bt_len * padded_vocab_size;
        let logits = ctx
            .new_tensor_view(rand_f32_vec(bt_padded_len).as_slice())
            .expect("tensor alloc failed");
        let losses =
            ctx.new_tensor_view(rand_f32_vec(bt_len).as_slice()).expect("tensor alloc failed");
        let dlosses =
            ctx.new_tensor_view(rand_f32_vec(bt_len).as_slice()).expect("tensor alloc failed");
        let targets = ctx
            .new_tensor_view(
                rand_i32_vec(bt_len)
                    .iter()
                    .map(|x| x.abs() % (padded_vocab_size as i32))
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .expect("tensor alloc failed");
        let empty = ctx.new_tensor_view([].as_slice()).unwrap();
        Some(Self { config, logits, losses, dlosses, targets, empty })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        const BSIZE: u32 = 1024;
        let grid_size = (self.config.batch_size * self.config.seq_len) as u32;
        gpu_host::gpu_config!(grid_size, 1, 1, @const BSIZE, 1, 1, 0)
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        llm_rs_gpu::fused_classifier_kernel3::launch(
            self.launch_config(),
            ctx,
            m,
            &mut self.logits,
            &mut self.losses,
            &mut self.empty,
            &self.dlosses,
            &self.targets,
            self.config.batch_size as _,
            self.config.seq_len as _,
            self.config.vocab_size as _,
            self.config.padded_vocab_size as _,
        )
        .expect("failed to launch fused_classifier_kernel3");
        /*llmrs::kernels::fused_classifier3(
            ctx,
            m,
            &mut self.logits,
            &mut self.losses,
            &mut self.dlosses,
            &self.targets,
            self.config.batch_size,
            self.config.seq_len,
            self.config.vocab_size,
            self.config.padded_vocab_size,
        );*/
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::fused_classifier_host(
                self.logits.as_devptr() as _,
                self.losses.as_devptr() as _,
                self.dlosses.as_devptr() as _,
                self.targets.as_devptr() as _,
                self.config.batch_size as _,
                self.config.seq_len as _,
                self.config.vocab_size as _,
                self.config.padded_vocab_size as _,
            );
        }
    }
}

gen_bench!(FusedClassifier, "fused_classifier");
