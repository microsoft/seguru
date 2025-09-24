use core::marker::Sized;

#[rustc_diagnostic_item = "gpu::printf"]
#[gpu_codegen::device]
#[inline(never)]
pub fn printf(_fmt: &'static str) {
    unimplemented!()
}

trait SupportedPrintfArg {
    const HOLDER: &'static str;
}

#[allow(private_bounds)]
pub trait PushPrintfArg: SupportedPrintfArg + Sized {
    /// Push an argument to the printf function.
    fn _push_printf_arg(self, _holder: &'static str);

    #[gpu_codegen::device]
    #[inline(always)]
    fn push_printf_arg(self) {
        self._push_printf_arg(Self::HOLDER);
    }
}

impl<T: SupportedPrintfArg + Sized> PushPrintfArg for T {
    #[rustc_diagnostic_item = "gpu::print_args"]
    #[gpu_codegen::device]
    #[inline(never)]
    fn _push_printf_arg(self, _holder: &'static str) {
        // This will be implemented by the macro below.
        unimplemented!()
    }
}

macro_rules! def_push_printf_arg {
    ($t:ty, $holder: literal) => {
        impl SupportedPrintfArg for $t {
            const HOLDER: &'static str = $holder;
        }
    };
    () => {};
}

def_push_printf_arg!(u8, "%c");
def_push_printf_arg!(u16, "%u");
def_push_printf_arg!(u32, "%u");
def_push_printf_arg!(u64, "%lu");
def_push_printf_arg!(u128, "%llu");
def_push_printf_arg!(i8, "%d");
def_push_printf_arg!(i16, "%d");
def_push_printf_arg!(i32, "%d");
def_push_printf_arg!(i64, "%ld");
def_push_printf_arg!(i128, "%lld");
def_push_printf_arg!(usize, "%ld");
def_push_printf_arg!(isize, "%ld");
def_push_printf_arg!(f32, "%f");
def_push_printf_arg!(f64, "%f");
def_push_printf_arg!(bool, "%d");

#[macro_export]
macro_rules! println {
    ($fmt:literal) => {{
        $crate::printf($fmt);
    }};
    ($fmt:literal, $($arg:expr),+ $(,)?) => {{
        use $crate::PushPrintfArg;
        $(
            ($arg).push_printf_arg();
        )+
        $crate::printf($fmt);
    }};
}

#[macro_export]
macro_rules! println_once {
    ($($any:tt)*) => {{
        if $crate::thread_id::<$crate::DimX>() == 0 && $crate::thread_id::<$crate::DimY>() == 0 && $crate::thread_id::<$crate::DimZ>() == 0 && $crate::block_id::<$crate::DimX>() == 0 && $crate::block_id::<$crate::DimY>() == 0 && $crate::block_id::<$crate::DimZ>() == 0 {
            $crate::println!($($any)*);
        }
    }}
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! debug_once {
    ($($any:tt)*) => {{
        println_once!($($any)*);
    }}
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! debug_once {
    ($($any:tt)*) => {{}};
}
