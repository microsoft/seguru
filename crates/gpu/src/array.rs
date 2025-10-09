pub trait ArrayReshape<const N: usize> {
    type Output: ?Sized;
    fn reshape(&self) -> &Self::Output;
}

/// Useful to optimize code with vector load/store.
/// If length of the slice is not a multiple of N,
/// the remaining elements will be ignored.
impl<const N: usize, T> ArrayReshape<N> for &[T] {
    type Output = [[T; N]];
    fn reshape(&self) -> &Self::Output {
        // SAFETY: the returned slice will be at same size or shorter, so it is safe.
        unsafe { &*core::ptr::slice_from_raw_parts_mut(self.as_ptr() as _, self.len() / N) }
    }
}
