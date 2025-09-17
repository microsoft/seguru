use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Expr, PatType, Stmt};

use crate::CodegenTarget;

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
    span: Span,
    target: CodegenTarget,
) {
    let is_gpu_only = target.is_gpu_only();
    let is_async = host_func.sig.asyncness.is_some();

    host_func.sig.output = syn::ReturnType::Type(
        syn::token::RArrow::default(),                                   // ->
        Box::new(syn::parse_quote! { Result<(), gpu_host::CudaError> }), // u32
    );

    let mut stmts = vec![];

    // 2. Build up argument list and add ctx, module, config to wrapper
    let wrapper_args = &host_func.sig.inputs;
    let host_args = syn::Ident::new("args_for_launching", span);
    let mod_arg = syn::Ident::new(if is_gpu_only { "_host_module" } else { "host_module" }, span);
    let ctx_arg = syn::Ident::new(if is_gpu_only { "_ctx" } else { "ctx" }, span);
    let config_arg = syn::Ident::new(if is_gpu_only { "_config" } else { "config" }, span);
    let ctx_type_arg = syn::Ident::new("CN", span);
    let config_type_arg = syn::Ident::new("Config", span);

    let args = wrapper_args
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
    stmts.push(Stmt::Expr(
        Expr::Verbatim(quote_spanned! {span =>let #host_args = [#(&#args as &dyn gpu_host::AsHostKernelParams),*];}),
        None,
    ));

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
        host_func.block.stmts.clear();
        let args = host_func
            .sig
            .inputs
            .iter()
            .skip(3)
            .map(|arg| {
                if let syn::FnArg::Typed(pat_type) = arg {
                    if let syn::Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                        pat_ident.ident.clone()
                    } else {
                        panic!("You don't have a name for your arg");
                    }
                } else {
                    panic!("Unexpected receiver argument");
                }
            })
            .collect::<Vec<_>>();
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
                config,
                &#host_args,
                None,
                #is_async,
            )
        }
    };

    stmts.push(Stmt::Expr(Expr::Verbatim(t), None));
    // Insert everything into the blocks of the wrapper
    host_func.block.stmts.clear();
    host_func.block.stmts.extend(stmts);
}

pub(crate) fn rewrite(attr: TokenStream, input: TokenStream, target: CodegenTarget) -> TokenStream {
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    let fun_span = fun.span();
    let kernel_fn_path = syn::parse_macro_input!(attr as syn::Path);

    // The newly generated function uses the same span as the attributes
    host_rewrite(&mut fun, kernel_fn_path.clone(), fun_span, target);
    fun.to_token_stream().into()
}
