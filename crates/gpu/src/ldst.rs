macro_rules! impl_ldcs {
    ($fname: ident, $ty:ty, $p: literal) => {
        #[gpu_codegen::device]
        #[inline(always)]
        pub fn $fname(ptr: &$ty) -> $ty {
            let mut ret: $ty;
            let ptr = ptr as *const $ty;
            unsafe {
                core::arch::asm!(
                    concat!("ld.global.cs.", $p, " {0:e}, [{1:r}];"),
                    out(reg) ret,
                    in(reg) ptr,
                );
            }
            ret
        }
    };
}

impl_ldcs!(__ldcs_u32, u32, "u32");
impl_ldcs!(__ldcs_i32, i32, "u32");
impl_ldcs!(__ldcs_f32, f32, "f32");

macro_rules! impl_stcs {
    ($fname: ident, $ty:ty, $p: literal) => {
        #[gpu_codegen::device]
        #[inline(always)]
        pub fn $fname(ptr: &mut $ty, val: $ty) {
            let ptr = ptr as *mut $ty;
            unsafe {
                core::arch::asm!(
                    concat!("st.global.cs.", $p, "[{0:r}], {1:e};"),
                    in(reg) ptr,
                    in(reg) val,
                );
            }
        }
    };
}

impl_ldcs!(__stcs_u32, u32, "u32");
impl_stcs!(__stcs_i32, i32, "u32");
impl_stcs!(__stcs_f32, f32, "f32");

pub trait CacheStreamLoadStore: Sized {
    type Output;
    fn ldcs(&self) -> Self::Output;
    fn stcs(&mut self, val: Self::Output);
}

impl CacheStreamLoadStore for i32 {
    type Output = i32;
    #[gpu_codegen::device]
    #[inline(always)]
    fn ldcs(&self) -> Self::Output {
        __ldcs_i32(self)
    }

    #[gpu_codegen::device]
    #[inline(always)]
    fn stcs(&mut self, val: Self::Output) {
        __stcs_i32(self, val)
    }
}

impl CacheStreamLoadStore for f32 {
    type Output = f32;

    #[gpu_codegen::device]
    #[inline(always)]
    fn ldcs(&self) -> Self::Output {
        __ldcs_f32(self)
    }

    #[gpu_codegen::device]
    #[inline(always)]
    fn stcs(&mut self, val: Self::Output) {
        __stcs_f32(self, val)
    }
}
