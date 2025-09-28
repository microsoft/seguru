pub(crate) trait SizedOrSlice: 'static {
    type UnitType: Sized;
    fn len_if_slice(&self) -> Option<usize>;
    fn build_const_ptr(ptr: *const u8, len: usize) -> *const Self;
}

impl<T: Sized + 'static> SizedOrSlice for T {
    type UnitType = T;
    fn len_if_slice(&self) -> Option<usize> {
        None
    }

    fn build_const_ptr(ptr: *const u8, len: usize) -> *const Self {
        assert_eq!(len, 1);
        ptr as *const Self
    }
}

impl<T: 'static> SizedOrSlice for [T] {
    type UnitType = T;
    fn len_if_slice(&self) -> Option<usize> {
        Some(self.len())
    }

    fn build_const_ptr(ptr: *const u8, len: usize) -> *const Self {
        core::ptr::slice_from_raw_parts(ptr as *const T, len)
    }
}

pub(crate) trait SizedOrSliceClone: SizedOrSlice {}

impl<T: SizedOrSlice + Clone> SizedOrSliceClone for T {}

impl<T: SizedOrSlice + Clone> SizedOrSliceClone for [T] {}
