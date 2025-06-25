extern crate proc_macro;
use proc_macro::TokenStream;
mod gpu_syntax;
mod kernel_rewriter;

#[proc_macro_attribute]
pub fn host(_: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_host(item)
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel_rewriter::rewrite(attr, item)
}
