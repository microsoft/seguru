#![allow(clippy::too_many_arguments)]
use syntax_gpu::kernel_arith as kernel_arith_gpu;

#[gpu::host(kernel_arith_gpu::<30>)]
pub fn kernel_arith(
    a: &gpu_host::TensorView<'_, [u32]>,
    b: &mut gpu_host::TensorViewMut<'_, [u32]>,
    c: &gpu_host::TensorView<'_, [u32]>,
    f: &mut gpu_host::TensorViewMut<'_, [f32]>,
    f_width: usize,
    g: &gpu_host::TensorView<'_, [f32]>,
    h: &mut gpu_host::TensorViewMut<'_, f32>,
) {
}
