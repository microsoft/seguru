extern crate proc_macro;
use proc_macro::TokenStream;
mod gpu_syntax;

#[proc_macro_attribute]
pub fn host(_: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_host(item)
}
