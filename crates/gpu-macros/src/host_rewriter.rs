use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Expr, PatType, Stmt};

use crate::CodegenTarget;

fn host_rewrite(
    host_func: &mut syn::ItemFn,
    kernel_fn_path: syn::Path,
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

    stmts.push(Stmt::Expr(
        Expr::Verbatim(quote_spanned! {span =>let mut #host_args = Vec::<&dyn gpu_host::AsHostKernelParams>::new();}),
        None,
    ));
    wrapper_args.iter().for_each(|arg| {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                stmts.push(Stmt::Expr(
                    Expr::Verbatim(quote_spanned! {span => #host_args.push(&#pat_ident);}),
                    None,
                ));
            } else {
                panic!("You don't have a name for your arg");
            }
        }
    });
    let mut bounds = syn::punctuated::Punctuated::new();
    bounds.push(syn::parse_quote! { gpu_host::GpuCtxSpace });
    bounds.push(syn::parse_quote! { 'static });
    host_func.sig.generics.params.push(syn::GenericParam::Type(syn::TypeParam {
        attrs: vec![],
        ident: syn::Ident::new("N", span),
        colon_token: Some(syn::Token![:](span)),
        bounds,
        eq_token: None,
        default: None,
    }));
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
            ty: Box::new(syn::parse_quote! { gpu_host::GPUConfig }),
        }),
    );
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
            ty: Box::new(syn::parse_quote! { &gpu_host::GpuModule<N> }),
        }),
    );
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
            ty: Box::new(syn::parse_quote! { &gpu_host::GpuCtxGuard<'_, '_, N> }),
        }),
    );

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
                #kernel_fn_path(#(#args,)*);
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
                0,
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
