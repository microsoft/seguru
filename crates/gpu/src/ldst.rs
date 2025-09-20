macro_rules! impl_ld {
    ($fname: ident, $ty:ty, $c: literal, $t: literal) => {
        #[gpu_codegen::device]
        #[inline(always)]
        pub fn $fname(ptr: &$ty) -> $ty {
            let mut ret: $ty;
            let ptr = ptr as *const $ty;
            unsafe {
                core::arch::asm!(
                    concat!("ld.global.", $c, ".", $t, " {0:e}, [{1:r}];"),
                    out(reg) ret,
                    in(reg) ptr,
                );
            }
            ret
        }
    };
}

impl_ld!(__ldcs_u32, u32, "cs", "u32");
impl_ld!(__ldcs_i32, i32, "cs", "u32");
impl_ld!(__ldcs_f32, f32, "cs", "f32");

impl_ld!(__ldcg_u32, u32, "cg", "u32");
impl_ld!(__ldcg_i32, i32, "cg", "u32");
impl_ld!(__ldcg_f32, f32, "cg", "f32");

macro_rules! impl_st {
    ($fname: ident, $ty:ty, $c: literal, $t: literal) => {
        #[gpu_codegen::device]
        #[inline(always)]
        pub fn $fname(ptr: &mut $ty, val: $ty) {
            let ptr = ptr as *mut $ty;
            unsafe {
                core::arch::asm!(
                    concat!("st.global.", $c, ".", $t, "[{0:r}], {1:e};"),
                    in(reg) ptr,
                    in(reg) val,
                );
            }
        }
    };
}

impl_st!(__stcs_u32, u32, "cs", "u32");
impl_st!(__stcs_i32, i32, "cs", "u32");
impl_st!(__stcs_f32, f32, "cs", "f32");

impl_st!(__stcg_u32, u32, "cg", "u32");
impl_st!(__stcg_i32, i32, "cg", "u32");
impl_st!(__stcg_f32, f32, "cg", "f32");

pub trait CacheStreamLoadStore: Sized {
    type Output;
    fn ldcs(&self) -> Self::Output;
    fn stcs(&mut self, val: Self::Output);
    fn ldcg(&self) -> Self::Output;
    fn stcg(&mut self, val: Self::Output);
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

    #[gpu_codegen::device]
    #[inline(always)]
    fn ldcg(&self) -> Self::Output {
        __ldcg_i32(self)
    }

    #[gpu_codegen::device]
    #[inline(always)]
    fn stcg(&mut self, val: Self::Output) {
        __stcg_i32(self, val)
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

    #[gpu_codegen::device]
    #[inline(always)]
    fn ldcg(&self) -> Self::Output {
        __ldcg_f32(self)
    }

    #[gpu_codegen::device]
    #[inline(always)]
    fn stcg(&mut self, val: Self::Output) {
        __stcg_f32(self, val)
    }
}
