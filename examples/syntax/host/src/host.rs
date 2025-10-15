#![allow(clippy::too_many_arguments)]
use syntax_gpu::kernel_arith as kernel_arith_gpu;

#[gpu::host(kernel_arith_gpu::<30>)]
pub fn kernel_arith(
    a: &gpu_host::TensorView<'_, [u32]>,
    b: gpu::GlobalThreadChunk<'_, u32, gpu::Map2D>,
    c: &gpu_host::TensorView<'_, [u32]>,
    f: &mut gpu_host::TensorViewMut<'_, [f32]>,
    f_width: usize,
    g: &gpu_host::TensorView<'_, [f32]>,
    h: &mut gpu_host::TensorViewMut<'_, f32>,
) {
}

#[gpu::host(syntax_gpu::oob1)]
pub fn oob1(a: f32, b: &mut gpu_host::TensorViewMut<'_, [f32]>, width: usize) {}

#[gpu::host(syntax_gpu::oob_no_fails)]
pub fn oob_no_fails(a: f32, b: &mut gpu_host::TensorViewMut<'_, [f32]>, width: usize) {}

#[gpu::host(syntax_gpu::oob2)]
pub fn oob2(a: f32, b: &mut gpu_host::TensorViewMut<'_, [f32]>, width: usize) {}

#[gpu::host(syntax_gpu::oob3)]
pub fn oob3(a: f32, b: &mut gpu_host::TensorViewMut<'_, [f32]>, width: usize) {}
