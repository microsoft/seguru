use crate::{DimType, assert_before_index, block_dim, block_id, dim, grid_dim, thread_id};
#[cfg(not(feature = "codegen_tests"))]
use crate::{GpuChunkable2D, GpuChunkableMut2D};

#[inline(always)]
#[gpu_codegen::device]
#[rustc_diagnostic_item = "gpu::chunk_mut"]
#[gpu_codegen::sync_data(0, 1)]
pub fn chunk_mut<T>(original: &mut [T], window: usize, idx: crate::GpuChunkIdx) -> &mut [T] {
    let offset = idx.as_usize() * window;
    // SAFETY: This is safe since GpuChunkIdx is unique per GPU thread.
    unsafe { crate::subslice_mut(original, offset, window) }
}

#[inline(always)]
#[gpu_codegen::device]
pub fn chunk<T>(original: &[T], window: usize, idx: crate::GpuChunkIdx) -> &[T] {
    let offset = idx.as_usize() * window;
    crate::subslice(original, offset, window)
}

#[cfg(not(feature = "codegen_tests"))]
#[rustc_diagnostic_item = "gpu::get_local_mut_2d"]
#[gpu_codegen::device]
pub fn get_local_mut_2d<'a, T>(a: &mut GpuChunkableMut2D<'a, T>, x: usize, y: usize) -> &'a mut T {
    // Must check if col is smaller than a.size_x

    let row =
        y * dim(DimType::Y) + block_dim(DimType::Y) * block_id(DimType::Y) + thread_id(DimType::Y);
    let col =
        x * dim(DimType::X) + block_dim(DimType::X) * block_id(DimType::X) + thread_id(DimType::X);
    let z_size = dim(DimType::Z);
    let idx = a.size_x() * row + col;
    assert_before_index(z_size == 1, idx);
    assert_before_index(col <= a.size_x(), idx);

    // Here Rust will automatic generate an SFI
    unsafe { &mut (&mut *(a.as_ptr() as *mut [T]))[a.size_x() * row + col] }
}

#[cfg(not(feature = "codegen_tests"))]
#[rustc_diagnostic_item = "gpu::get_local_2d"]
#[gpu_codegen::device]
pub fn get_local_2d<'a, T>(a: &'a GpuChunkable2D<'a, T>, x: usize, y: usize) -> &'a T {
    // Must check if col is smaller than a.size_x

    use crate::assert_before_index;
    let row = y * grid_dim(DimType::Y) * block_dim(DimType::Y)
        + block_dim(DimType::Y) * block_id(DimType::Y)
        + thread_id(DimType::Y);
    let col = x * grid_dim(DimType::X) * block_dim(DimType::X)
        + block_dim(DimType::X) * block_id(DimType::X)
        + thread_id(DimType::X);
    let z_size = grid_dim(DimType::Z) * block_dim(DimType::Z);
    let idx = a.size_x() * row + col;
    assert_before_index(z_size == 1, idx);
    assert_before_index(col <= a.size_x(), idx);

    // Here Rust will automatic generate an SFI
    unsafe { &(&*a.as_ptr())[idx] }
}
