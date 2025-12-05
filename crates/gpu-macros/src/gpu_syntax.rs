use proc_macro::TokenStream;
use quote::ToTokens;
use syn::parse::Parse;
use syn::visit_mut::VisitMut;
use syn::{Path, parse_quote};

#[derive(Clone, Debug)]
pub(crate) struct KernelAttr {
    pub dynamic_shared: bool,
}

impl Parse for KernelAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let args =
            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated(input)?;
        let mut dynamic_shared = false;
        for arg in args {
            match arg {
                syn::Meta::Path(p) if p.is_ident("dynamic_shared") => dynamic_shared = true,
                _ => {
                    return Err(syn::Error::new_spanned(arg, "Unknown attribute argument"));
                }
            }
        }
        Ok(KernelAttr { dynamic_shared })
    }
}

pub struct GpuFunctionRewriter {
    pub(crate) is_kernel_entry: bool,
    pub(crate) target: crate::CodegenTarget,
    pub(crate) kernel_attr: KernelAttr,
}

impl GpuFunctionRewriter {
    pub fn new(
        is_kernel_entry: bool,
        kernel_attr: KernelAttr,
        target: crate::CodegenTarget,
    ) -> Self {
        GpuFunctionRewriter { is_kernel_entry, kernel_attr, target }
    }
}

fn path_matches(path: &Path, target: &str) -> bool {
    let segs = target.split("::").collect::<Vec<_>>();
    let segments: Vec<_> = path.segments.iter().map(|seg| seg.ident.to_string()).collect();
    segments == segs
}

impl VisitMut for GpuFunctionRewriter {
    fn visit_fn_arg_mut(&mut self, arg: &mut syn::FnArg) {
        // Replace &'a mut T args with GpuGlobal<'a, T>
        // kernel entry function is guaranteed to use global memory for &mut T args
        if !self.is_kernel_entry {
            // Only modify the arguments of kernel functions
            return;
        }
        if let syn::FnArg::Typed(pat_type) = arg
            && let syn::Type::Reference(type_ref) = &*pat_type.ty
            && type_ref.mutability.is_some()
        {
            let inner_type = &*type_ref.elem;
            let lifetime = type_ref.lifetime.as_ref().map_or(parse_quote!('_), |lt| lt.clone());
            let new_type: syn::Type = parse_quote! {
                ::gpu::GpuGlobal<#lifetime, #inner_type>
            };
            *pat_type.ty = new_type;
        }
    }
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
            }
        }
        if self.kernel_attr.dynamic_shared {
            f.sig.inputs.push(parse_quote!(mut smem_alloc: ::gpu::DynamicSharedAlloc));
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
    kernel_attr: KernelAttr,
    target: crate::CodegenTarget,
) {
    let mut dev_rewriter =
        crate::gpu_syntax::GpuFunctionRewriter::new(is_kernel_entry, kernel_attr, target);
    dev_rewriter.visit_item_fn_mut(fun);
}

pub(crate) fn rewrite_gpu_code(
    attr: TokenStream,
    input: TokenStream,
    is_kernel_entry: bool,
    target: crate::CodegenTarget,
) -> TokenStream {
    let kernel_attr = syn::parse_macro_input!(attr as KernelAttr);
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    basic_rewrite_gpu_func(&mut fun, is_kernel_entry, kernel_attr, target);
    fun.to_token_stream().into()
}
