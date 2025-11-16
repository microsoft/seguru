#[macro_export]
macro_rules! asm {
    ($asm:expr, $($operands:tt)*) => {{
        core::arch::asm!($crate::nvptx_to_target_asm!($asm), $($operands)*);
    }};
    ($asm:expr) => {{
        core::arch::asm!($crate::nvptx_to_target_asm!($asm));
    }};
}
