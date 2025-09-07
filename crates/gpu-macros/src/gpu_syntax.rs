use proc_macro::TokenStream;
use quote::ToTokens;
use syn::visit_mut::VisitMut;
use syn::{Path, parse_quote};

pub struct GpuFunctionRewriter {
    pub(crate) is_kernel_entry: bool,
    pub(crate) target: crate::CodegenTarget,
}

impl GpuFunctionRewriter {
    pub fn new(is_kernel_entry: bool, target: crate::CodegenTarget) -> Self {
        GpuFunctionRewriter { is_kernel_entry, target }
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
        if !self.target.need_register_tool() {
            return;
        }
        if !closure.attrs.iter().any(|a| path_matches(a.path(), "gpu_codegen::device")) {
            closure.attrs.push(parse_quote!(#[gpu_codegen::device]));
        }
    }

    fn visit_item_fn_mut(&mut self, f: &mut syn::ItemFn) {
        syn::visit_mut::visit_item_fn_mut(self, f);
        #[cfg(not(feature = "codegen_tests"))]
        {
            if self.is_kernel_entry {
                // Add dynamic config params
                f.sig.generics.params.insert(0, syn::parse_quote! { Config: ::gpu::SafeGpuConfig});
                f.block.stmts.push(parse_quote! {
                    unsafe {::gpu::assume_dim_with_config::<Config>();}
                });
            }
        }
        if !self.target.need_register_tool() {
            return;
        }
        if self.is_kernel_entry {
            // If we are inside a host function, we need to ensure it has the #[gpu_codegen::host] attribute
            if !f.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::kernel")) {
                f.attrs.push(parse_quote!(#[gpu_codegen::kernel]));
            }
        } else {
            // If we are inside a device function, we need to ensure it has the #[gpu_codegen::device] attribute
            if !f.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::device")) {
                f.attrs.push(parse_quote!(#[gpu_codegen::device]));
            }
        }
    }
}

pub(crate) fn basic_rewrite_gpu_func(
    fun: &mut syn::ItemFn,
    is_kernel_entry: bool,
    target: crate::CodegenTarget,
) {
    if target.erase_func_body() {
        fun.attrs.push(parse_quote!(#[allow(unused_variables)]));
        fun.block.stmts.clear();
        fun.block.stmts.push(parse_quote! {
            unimplemented!();
        });
    }
    let mut dev_rewriter = crate::gpu_syntax::GpuFunctionRewriter::new(is_kernel_entry, target);
    dev_rewriter.visit_item_fn_mut(fun);
}

pub(crate) fn rewrite_gpu_code(
    _: TokenStream,
    input: TokenStream,
    is_kernel_entry: bool,
    target: crate::CodegenTarget,
) -> TokenStream {
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    basic_rewrite_gpu_func(&mut fun, is_kernel_entry, target);
    fun.to_token_stream().into()
}
