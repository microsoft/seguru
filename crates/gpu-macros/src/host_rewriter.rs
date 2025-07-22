use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Expr, PatType, Stmt};

fn host_rewrite(host_func: &mut syn::ItemFn, span: Span) {
    // Rewrite:
    // 1. Change to return error
    let kernel_func_str = &format!("{}", &host_func.sig.ident);

    host_func.attrs.clear();
    let is_async = host_func.sig.asyncness.is_some();
    host_func.sig.output = syn::ReturnType::Type(
        syn::token::RArrow::default(), // ->
        Box::new(syn::parse_quote! { Result<(), cuda_bindings::CudaError> }), // u32
    );

    let mut stmts = vec![];

    // 2. Build up argument list and add ctx, module, config to wrapper
    let wrapper_args = &host_func.sig.inputs;
    let host_args = syn::Ident::new("args_for_launching", span);
    let mod_arg = syn::Ident::new("host_module", span);
    let ctx_arg = syn::Ident::new("ctx", span);
    let config_arg = syn::Ident::new("config", span);

    stmts.push(Stmt::Expr(
        Expr::Verbatim(quote_spanned! {span =>let mut #host_args = Vec::<&dyn cuda_bindings::AsHostKernelParams>::new();}),
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
    bounds.push(syn::parse_quote! { cuda_bindings::GpuCtxSpace });
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
            ty: Box::new(syn::parse_quote! { cuda_bindings::GPUConfig }),
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
            ty: Box::new(syn::parse_quote! { &cuda_bindings::GpuModule<N> }),
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
            ty: Box::new(syn::parse_quote! { &cuda_bindings::GpuCtxGuard<'_, '_, N> }),
        }),
    );

    // 3. Call the actual function
    let shared_mem_size_ident =
        syn::Ident::new(&format!("const_share_size_{}", kernel_func_str), span);
    let t = quote_spanned! {span =>
        // #Safety: this is safe if the argument types match the kernel's expected types.
        unsafe {
            #ctx_arg.launch_kernel(
                #mod_arg,
                #kernel_func_str,
                config,
                #shared_mem_size_ident as _,
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

pub(crate) fn rewrite(attr: TokenStream, input: TokenStream) -> TokenStream {
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    let fun_span = fun.span();

    //println!("{:?}", attr.into_iter().last());

    let mut wrapper_stream = proc_macro2::TokenStream::new();

    // Add the constant shared mem thing
    let func_ident: proc_macro::Ident =
        if let Some(proc_macro::TokenTree::Ident(ident)) = attr.clone().into_iter().last() {
            ident
        } else {
            panic!("Your must pass your kernel function's ident");
        };
    let local_share_mem_size_ident =
        syn::Ident::new(&format!("const_share_size_{}", func_ident), fun_span);
    let mut const_val_def: proc_macro::TokenStream =
        quote! { #[allow(non_upper_case_globals)] const #local_share_mem_size_ident: usize = }
            .into();
    let mut attr_token_trees: Vec<proc_macro::TokenTree> = attr.into_iter().collect();
    attr_token_trees.pop();
    let share_mem_prefix: proc_macro::TokenStream = attr_token_trees.into_iter().collect();
    const_val_def.extend(share_mem_prefix);
    let share_mem_size_ident = syn::Ident::new(&format!("shared_size_{}", func_ident), fun_span);
    let const_val_def_end: proc_macro::TokenStream = quote! { #share_mem_size_ident ; }.into();
    const_val_def.extend(const_val_def_end);

    // The newly generated function uses the same span as the attributes
    host_rewrite(&mut fun, fun_span);
    fun.to_tokens(&mut wrapper_stream);

    let wrapper_stream_macro: proc_macro::TokenStream = wrapper_stream.into();
    const_val_def.extend(wrapper_stream_macro);

    // Original function is dropped and replaced with our stuff

    // println!("{}", const_val_def);

    const_val_def
}
