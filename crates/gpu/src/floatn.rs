pub type Float4 = Float32N<4>;

#[rustc_diagnostic_item = "gpu::Float32N"]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Float32N<const N: usize> {
    pub data: [f32; N],
}

impl<const N: usize> Default for Float32N<N> {
    fn default() -> Self {
        Self { data: [0.0; N] }
    }
}

impl<const N: usize> Float32N<N> {
    #[inline(always)]
    pub fn new(data: [f32; N]) -> Self {
        Self { data }
    }
}

impl<const N: usize> core::ops::Add for Float32N<N> {
    type Output = Self;
    /// Adds two vectors element-wise.
    #[gpu_codegen::device]
    #[rustc_diagnostic_item = "gpu::Float32N::add"]
    fn add(self, rhs: Self) -> Self {
        let mut data = [0.0; N];
        for ((out, lhs), rhs) in data.iter_mut().take(N).zip(self.data.iter()).zip(rhs.data.iter())
        {
            *out = *lhs + *rhs;
        }
        Self { data }
    }
}
