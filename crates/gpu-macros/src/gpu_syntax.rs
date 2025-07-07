use proc_macro::TokenStream;
use quote::ToTokens;
use syn::visit_mut::VisitMut;
use syn::{Path, parse_quote};

pub struct GpuFunctionRewriter {
    pub(crate) is_kernel_entry: bool,
    pub(crate) is_device: bool,
}

impl GpuFunctionRewriter {
    pub fn new(is_kernel_entry: bool, is_device: bool) -> Self {
        GpuFunctionRewriter { is_kernel_entry, is_device }
    }
}

fn path_matches(path: &Path, target: &str) -> bool {
    let segs = target.split("::").collect::<Vec<_>>();
    let segments: Vec<_> = path.segments.iter().map(|seg| seg.ident.to_string()).collect();
    segments == segs
}

impl VisitMut for GpuFunctionRewriter {
    fn visit_expr_closure_mut(&mut self, closure: &mut syn::ExprClosure) {
        syn::visit_mut::visit_expr_closure_mut(self, closure);
        if !closure.attrs.iter().any(|a| path_matches(a.path(), "gpu_codegen::device")) {
            closure.attrs.push(parse_quote!(#[gpu_codegen::device]));
        }
    }

    fn visit_item_fn_mut(&mut self, f: &mut syn::ItemFn) {
        syn::visit_mut::visit_item_fn_mut(self, f);
        assert!(
            !self.is_kernel_entry || !self.is_device,
            "Cannot be both kernel entry and device function"
        );
        if self.is_kernel_entry {
            // If we are inside a host function, we need to ensure it has the #[gpu_codegen::host] attribute
            if !f.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::kernel")) {
                f.attrs.push(parse_quote!(#[gpu_codegen::kernel]));
                f.attrs.push(parse_quote!(#[unsafe(no_mangle)]));
            }
        } else if self.is_device {
            // If we are inside a device function, we need to ensure it has the #[gpu_codegen::device] attribute
            if !f.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::device")) {
                f.attrs.push(parse_quote!(#[gpu_codegen::device]));
            }
        }
    }
}

pub(crate) fn basic_rewrite_device(fun: &mut syn::ItemFn) {
    let mut dev_rewriter = crate::gpu_syntax::GpuFunctionRewriter::new(false, true);
    dev_rewriter.visit_item_fn_mut(fun);
}

pub(crate) fn basic_rewrite_kernel_entry(fun: &mut syn::ItemFn) {
    let mut entry_rewriter = crate::gpu_syntax::GpuFunctionRewriter::new(true, false);
    entry_rewriter.visit_item_fn_mut(fun);
}

pub(crate) fn rewrite_gpu_code(
    _: TokenStream,
    input: TokenStream,
    is_kernel_entry: bool,
    is_gpu_code: bool,
) -> TokenStream {
    if !is_gpu_code {
        return input; // If not GPU code, return the original input
    }
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    if is_kernel_entry {
        basic_rewrite_kernel_entry(&mut fun);
    } else {
        basic_rewrite_device(&mut fun);
    }
    fun.to_token_stream().into()
}

pub(crate) fn rewrite_shared_size(input: TokenStream) -> TokenStream {
    let mut static_def = syn::parse_macro_input!(input as syn::ItemStatic);

    if !static_def.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::shared_size")) {
        static_def.attrs.push(parse_quote!(#[gpu_codegen::shared_size]));
        static_def.attrs.push(parse_quote!(#[allow(non_upper_case_globals)]));
    }

    let source_code = static_def.to_token_stream().to_string();
    println!("{}", source_code);

    static_def.to_token_stream().into()
}
