/// A GPU iterator over a slice of elements.
/// The Iterator implementation in core crate is not optimized for GPU.
/// It is recommended to use the provided GPU iterator abstractions for better performance.
pub struct GpuIter<'a, T: 'a> {
    /// The pointer to the next element to return, or the past-the-end location
    /// if the iterator is empty.
    ///
    /// This address will be used for all ZST elements, never changed.
    slice: &'a [T],
    /// For non-ZSTs, the non-null pointer to the past-the-end element.
    ///
    /// For ZSTs, this is `ptr::without_provenance_mut(len)`.
    idx: usize,
}

impl<'a, T: 'a> GpuIter<'a, T> {
    #[inline(always)]
    #[gpu_codegen::device]
    pub fn new(s: &'a [T]) -> Self {
        Self { slice: s, idx: 0 }
    }
}

impl<'a, T: 'a> Iterator for GpuIter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn next(&mut self) -> Option<Self::Item> {
        let ret = if self.idx < self.slice.len() {
            let item = unsafe { self.slice.get_unchecked(self.idx) };
            Some(item)
        } else {
            None
        };
        self.idx += 1;
        ret
    }
}
