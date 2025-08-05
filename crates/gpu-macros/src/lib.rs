#![feature(box_patterns)]

extern crate proc_macro;

use proc_macro::TokenStream;
mod gpu_syntax;
mod host_rewriter;
mod kernel_rewriter;

fn is_gpu_code() -> bool {
    #[cfg(gpu_codegen)]
    {
        true
    }
    #[cfg(not(gpu_codegen))]
    {
        false
    }
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel_rewriter::rewrite(attr, item, is_gpu_code())
}

#[proc_macro_attribute]
pub fn host(attr: TokenStream, item: TokenStream) -> TokenStream {
    host_rewriter::rewrite(attr, item)
}

#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_gpu_code(attr, item, false, is_gpu_code())
}

#[proc_macro_attribute]
pub fn kernel_v2(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_gpu_code(attr, item, true, is_gpu_code())
}
