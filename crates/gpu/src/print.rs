use core::marker::Sized;

#[rustc_diagnostic_item = "gpu::printf"]
#[gpu_codegen::device]
#[inline(never)]
pub fn printf() -> usize {
    unimplemented!()
}

trait SupportedPrintfArg {}

#[allow(private_bounds)]
pub trait PushPrintfArg: SupportedPrintfArg + Sized {
    /// Push an argument to the printf function.
    fn push_printf_arg(self);
}

impl<T: SupportedPrintfArg + Sized> PushPrintfArg for T {
    #[rustc_diagnostic_item = "gpu::print_args"]
    #[gpu_codegen::device]
    #[inline(never)]
    fn push_printf_arg(self) {
        // This will be implemented by the macro below.
        unimplemented!()
    }
}

macro_rules! def_push_printf_arg {
    ($t:ty) => {
        impl SupportedPrintfArg for $t {}
    };
    () => {};
}

def_push_printf_arg!(u8);
def_push_printf_arg!(u16);
def_push_printf_arg!(u32);
def_push_printf_arg!(u64);
def_push_printf_arg!(i8);
def_push_printf_arg!(i16);
def_push_printf_arg!(i32);
def_push_printf_arg!(i64);
def_push_printf_arg!(usize);
def_push_printf_arg!(isize);
def_push_printf_arg!(f32);
def_push_printf_arg!(f64);
def_push_printf_arg!(bool);

#[macro_export]
macro_rules! println {
    ($fmt:literal) => {{
        gpu::add_mlir_string_attr(concat!("\"", $fmt, "\""));
        gpu::printf();
    }};
    ($fmt:literal, $($arg:expr),+ $(,)?) => {{
        gpu::add_mlir_string_attr(concat!("\"", $fmt, "\""));
        use gpu::PushPrintfArg;
        $(
            ($arg).push_printf_arg();
        )+
        gpu::printf();
    }};
}
