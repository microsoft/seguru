use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Expr, PatType, Stmt};

use crate::CodegenTarget;
use crate::gpu_syntax::KernelAttr;

fn generic_ctxspace_bound(ident: syn::Ident, has_default: bool, span: Span) -> syn::GenericParam {
    let mut bounds = syn::punctuated::Punctuated::new();
    bounds.push(syn::parse_quote! { ::gpu_host::GpuCtxSpace });
    bounds.push(syn::parse_quote! { 'static });
    syn::GenericParam::Type(syn::TypeParam {
        attrs: vec![],
        ident,
        colon_token: Some(syn::Token![:](span)),
        bounds,
        eq_token: if has_default { Some(syn::Token![=](span)) } else { None },
        default: if has_default {
            Some(syn::parse_quote! { ::gpu_host::CtxSpaceZero })
        } else {
            None
        },
    })
}

fn generic_config_bound(ident: syn::Ident, has_default: bool, span: Span) -> syn::GenericParam {
    let mut bounds = syn::punctuated::Punctuated::new();
    bounds.push(syn::parse_quote! { ::gpu_host::GPUConfig });
    syn::GenericParam::Type(syn::TypeParam {
        attrs: vec![],
        ident,
        colon_token: Some(syn::Token![:](span)),
        bounds,
        eq_token: if has_default { Some(syn::Token![=](span)) } else { None },
        default: if has_default {
            Some(syn::parse_quote! { ::gpu_host::GPUStaticConfig })
        } else {
            None
        },
    })
}

/// generate a host function
/*
fn kernel<..., Config: gpu_host::GPUConfig, C: gpu_host::GpuCtxSpace>(
    config: Config,
    ctx: &::gpu_host::GpuCtxGuard<'_, '_, C>,
    module: &::gpu_host::GpuModule<C>,
    args: Vec<&dyn gpu_host::AsHostKernelParams>,
) -> Result<(), gpu_host::CudaError> {
    Ok(())
}
*/
fn host_rewrite(
    host_func: &mut syn::ItemFn,
    mut kernel_fn_path: syn::Path,
    smem_alloc: bool,
    span: Span,
    target: CodegenTarget,
) {
    let is_gpu_only = target.is_gpu_only();
    let is_async = host_func.sig.asyncness.is_some();

    host_func.sig.output = syn::ReturnType::Type(
        syn::token::RArrow::default(),                                   // ->
        Box::new(syn::parse_quote! { Result<(), gpu_host::CudaError> }), // u32
    );

    host_func.block.stmts.clear();

    // 2. Build up argument list and add ctx, module, config to wrapper
    let wrapper_args = &host_func.sig.inputs;
    let host_args = syn::Ident::new("args_for_launching", span);
    let mod_arg = syn::Ident::new(if is_gpu_only { "_host_module" } else { "host_module" }, span);
    let ctx_arg = syn::Ident::new(if is_gpu_only { "_ctx" } else { "ctx" }, span);
    let config_arg = syn::Ident::new(if is_gpu_only { "_config" } else { "config" }, span);
    let ctx_type_arg = syn::Ident::new("CN", span);
    let config_type_arg = syn::Ident::new("Config", span);

    let mut args = wrapper_args
        .iter()
        .map(|arg| {
            if let syn::FnArg::Typed(pat_type) = arg {
                if let syn::Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                    pat_ident.ident.clone()
                } else {
                    panic!("You don't have a name for your arg");
                }
            } else {
                panic!("Unexpected untyped argument");
            }
        })
        .collect::<Vec<_>>();
    if smem_alloc {
        let smem_ident = syn::Ident::new("smem_alloc", span);
        host_func.block.stmts.push(Stmt::Expr(
            Expr::Verbatim(quote!(
                let #smem_ident = ::gpu::DynamicSharedAllocBuilder::smem_alloc(&#config_arg);
            )),
            None,
        ));
        args.push(syn::parse_quote!(#smem_ident));
    }

    // Add generic param for context namespace.
    host_func.sig.generics.params.push(generic_config_bound(config_type_arg.clone(), false, span));
    host_func.sig.generics.params.push(generic_ctxspace_bound(ctx_type_arg.clone(), false, span));
    // Add module arg
    host_func.sig.inputs.insert(
        0,
        syn::FnArg::Typed(PatType {
            attrs: vec![],
            pat: Box::new(syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident: mod_arg.clone(),
                subpat: None,
            })),
            colon_token: syn::token::Colon::default(),
            ty: Box::new(syn::parse_quote! { &::gpu_host::GpuModule<#ctx_type_arg> }),
        }),
    );

    // Add ctx arg
    host_func.sig.inputs.insert(
        0,
        syn::FnArg::Typed(PatType {
            attrs: vec![],
            pat: Box::new(syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident: ctx_arg.clone(),
                subpat: None,
            })),
            colon_token: syn::token::Colon::default(),
            ty: Box::new(syn::parse_quote! { &::gpu_host::GpuCtxGuard<'_, '_, #ctx_type_arg> }),
        }),
    );

    // Add dynamic config params
    host_func.sig.inputs.insert(
        0,
        syn::FnArg::Typed(PatType {
            attrs: vec![],
            pat: Box::new(syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident: config_arg.clone(),
                subpat: None,
            })),
            colon_token: syn::token::Colon::default(),
            ty: Box::new(syn::parse_quote! { #config_type_arg }),
        }),
    );

    let last_segment = kernel_fn_path.segments.last_mut().unwrap();
    if let syn::PathArguments::AngleBracketed(bracketed_args) = &mut last_segment.arguments {
        bracketed_args.args.insert(0, syn::parse_quote! { #config_type_arg });
    } else {
        last_segment.arguments =
            syn::PathArguments::AngleBracketed(syn::parse_quote! { ::<#config_type_arg> });
    }

    if target.is_gpu_only() {
        host_func.attrs.push(syn::parse_quote! { #[allow(clippy::extra_unused_type_parameters)] });
        // Add call to kernel function so that gpu2gpu will be able to monomorphize the kernel.
        host_func.block.stmts.push(Stmt::Expr(
            Expr::Verbatim(quote!(
                #kernel_fn_path(#(::gpu::HostToDev::convert(#args),)*);
            )),
            None,
        ));
        host_func.block.stmts.push(Stmt::Expr(Expr::Verbatim(quote!(Ok(()))), None));
        return;
    }

    let t = quote_spanned! {span =>
        // #Safety: this is safe if the argument types match the kernel's expected types.
        unsafe {
            #ctx_arg.launch_kernel(
                #mod_arg,
                &::gpu_host::get_fn_name(#kernel_fn_path),
                #config_arg,
                &#host_args,
                None,
                #is_async,
            )
        }
    };

    host_func.block.stmts.push(Stmt::Expr(
        Expr::Verbatim(quote_spanned! {span =>let #host_args = [#(&#args as &dyn gpu_host::AsHostKernelParams),*];}),
        None,
    ));
    host_func.block.stmts.push(Stmt::Expr(Expr::Verbatim(t), None));
}

pub(crate) fn rewrite(attr: TokenStream, input: TokenStream, target: CodegenTarget) -> TokenStream {
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    let fun_span = fun.span();
    let kernel_fn_path = syn::parse_macro_input!(attr as syn::Path);

    // The newly generated function uses the same span as the attributes
    host_rewrite(&mut fun, kernel_fn_path.clone(), false, fun_span, target);
    fun.to_token_stream().into()
}

pub(crate) fn create_host_from_kernel(
    attr: TokenStream,
    input: TokenStream,
    target: CodegenTarget,
) -> TokenStream {
    let kernel_attr = syn::parse_macro_input!(attr as KernelAttr);
    let mut kfun = syn::parse_macro_input!(input as syn::ItemFn);
    let kname = kfun.sig.ident.clone();
    let mut fun = kfun.clone();
    crate::gpu_syntax::basic_rewrite_gpu_func(&mut kfun, true, kernel_attr.clone(), target);
    fun.block.stmts.clear();
    fun.vis = syn::parse_quote!(pub);
    fun.sig.ident = syn::Ident::new("launch", kname.span());
    let kernel_fn_path = syn::parse_quote! {super::#kname};
    fun.sig.inputs.iter_mut().for_each(|arg| {
        if let syn::FnArg::Receiver(_) = arg {
            panic!("Kernel function cannot have receiver argument");
        }
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Type::Reference(type_ref) = pat_type.ty.as_mut() {
                let inner_type = &*type_ref.elem;
                type_ref.elem = if type_ref.mutability.is_some() {
                    syn::parse_quote! {
                            ::gpu_host::TensorViewMut<#inner_type>
                    }
                } else {
                    syn::parse_quote! {
                        ::gpu_host::TensorView<#inner_type>
                    }
                };
            }
        }
    });
    let span = fun.span();
    host_rewrite(&mut fun, kernel_fn_path, kernel_attr.dynamic_shared, span, target);
    quote_spanned! {span =>
            #kfun
            pub mod #kname {
                #[allow(unused_imports)] use super::*;
                #fun
            }
    }
    .into()
}
