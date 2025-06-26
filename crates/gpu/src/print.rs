#[gpu_codegen::builtin(gpu.printf)]
pub fn printf() -> usize {
    unimplemented!()
}

pub trait PushPrintfArg {
    /// Push an argument to the printf function.
    fn push_printf_arg(self);
}

macro_rules! def_push_printf_arg {
    ($t:ty) => {
        impl PushPrintfArg for $t {
            #[gpu_codegen::builtin(gpu.printf)]
            #[gpu_codegen::device]
            #[inline(never)]
            fn push_printf_arg(self) {
                unimplemented!()
            }
        }
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
