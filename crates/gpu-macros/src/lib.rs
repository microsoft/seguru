#![feature(box_patterns)]

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::ToTokens;
mod gpu_syntax;
mod host_rewriter;
mod reshape_map;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum CodegenTarget {
    Cpu,
    Gpu,
    GpuClippy,
}

impl CodegenTarget {
    fn is_gpu_only(&self) -> bool {
        matches!(self, CodegenTarget::Gpu)
    }

    fn need_register_tool(&self) -> bool {
        matches!(self, CodegenTarget::Gpu)
    }
}

fn target() -> CodegenTarget {
    let target = std::env::var("__CODEGEN_TARGET__").unwrap_or_else(|_| "CPU".into());
    match target.as_str() {
        "CPU" => CodegenTarget::Cpu,
        "GPU" => CodegenTarget::Gpu,
        "GPU-CLIPPY" => CodegenTarget::GpuClippy,
        _ => panic!("Unexpected __CODEGEN_TARGET__: {}", target),
    }
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_gpu_code(attr, item, true, target())
}

/// This attribute generates a host wrapper around a kernel function, allowing it to be launched from the host.
/// The kernel function itself is original function with Config.
/// The generated host function is in mod #kname {pub fn launch(...)}
#[proc_macro_attribute]
pub fn cuda_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    host_rewriter::create_host_from_kernel(attr, item, target())
}

#[proc_macro_attribute]
pub fn host(attr: TokenStream, item: TokenStream) -> TokenStream {
    host_rewriter::rewrite(attr, item, target())
}

#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_syntax::rewrite_gpu_code(attr, item, false, target())
}

#[proc_macro_attribute]
pub fn attr(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut kfun = syn::parse_macro_input!(item as syn::ItemFn);
    let attr = syn::parse_macro_input!(attr as syn::Ident);
    if target().need_register_tool() {
        kfun.attrs.push(syn::parse_quote!(#[gpu_codegen::#attr]));
    }
    kfun.into_token_stream().into()
}

#[proc_macro]
pub fn reshape_map(input: TokenStream) -> TokenStream {
    reshape_map::reshape_map(input)
}
