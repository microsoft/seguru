extern crate proc_macro;
use proc_macro::TokenStream;
mod gpu_syntax;
mod kernel_rewriter;

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel_rewriter::rewrite(attr, item)
}

#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_gpu_code(attr, item, false)
}

#[proc_macro_attribute]
pub fn kernel_v2(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_gpu_code(attr, item, true)
}
