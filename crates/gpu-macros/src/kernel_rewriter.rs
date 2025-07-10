use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::spanned::Spanned;

fn kernel_create_wrapper(func: &mut syn::ItemFn, span: Span) -> syn::ItemFn {
    // Rewrite:
    // 1. Create a new function and change its name and attributes
    let mut wrapper_func = func.clone();
    wrapper_func.sig.ident = syn::Ident::new(&format!("{}_wrapper", &func.sig.ident), func.span());

    wrapper_func.attrs.clear();
    wrapper_func.attrs.push(syn::parse_quote! {#[unsafe(no_mangle)]});

    let mut stmts = vec![
        syn::parse(
            quote! {
                let __c = gpu::block_dim(gpu::DimType::X) * gpu::block_id(gpu::DimType::X) + gpu::thread_id(gpu::DimType::X);
            }
            .into(),
        )
        .expect("Failed to parse input as a statement 2"),
    ];

    // 2. Add window to the argument list (also build the slices along the way)
    let wrapper_args = &mut wrapper_func.sig.inputs;
    let original_args = &mut func.sig.inputs;
    let mut call_args_rev = vec![];

    // Backwards iteration so that insertion don't mess up the indices
    // Thanks the good idea of the LLM of Google Search
    for i in (0..wrapper_args.len()).rev() {
        let arg = &mut wrapper_args[i];
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                if let syn::Type::Reference(type_ref) = &*pat_type.ty {
                    if let syn::Type::Path(path) = &*type_ref.elem {
                        // It MUST follow the gpu::GpuChunkable(Mut)

                        if path.path.segments[0].ident == "gpu"
                            && (path.path.segments[1].ident == "GpuChunkableMut"
                                || path.path.segments[1].ident == "GpuChunkable")
                        {
                            if let syn::PathArguments::AngleBracketed(slice_arg) =
                                &path.path.segments[1].arguments
                            {
                                if let syn::GenericArgument::Type(slice_ty) = &slice_arg.args[0] {
                                    let is_mut = path.path.segments[1].ident == "GpuChunkableMut";
                                    let replaced_arg_name = syn::Pat::Ident(syn::PatIdent {
                                        attrs: Vec::new(),
                                        by_ref: None,
                                        mutability: None,
                                        ident: syn::Ident::new(&pat_ident.ident.to_string(), span),
                                        subpat: None,
                                    });
                                    let replaced_arg = if is_mut {
                                        syn::FnArg::Typed(syn::PatType {
                                            attrs: Vec::new(),
                                            pat: Box::new(replaced_arg_name.clone()),
                                            colon_token: syn::token::Colon { spans: [span] },
                                            ty: Box::new(syn::Type::Verbatim(
                                                quote! { &mut [ #slice_ty ] },
                                            )),
                                        })
                                    } else {
                                        syn::FnArg::Typed(syn::PatType {
                                            attrs: Vec::new(),
                                            pat: Box::new(replaced_arg_name.clone()),
                                            colon_token: syn::token::Colon { spans: [span] },
                                            ty: Box::new(syn::Type::Verbatim(
                                                quote! { &[ #slice_ty ] },
                                            )),
                                        })
                                    };
                                    let arg_type = if is_mut {
                                        syn::Type::Verbatim(quote! { &mut [ #slice_ty ] })
                                    } else {
                                        syn::Type::Verbatim(quote! { &[ #slice_ty ] })
                                    };
                                    let arg_name = pat_ident.ident.clone();
                                    let window_arg_name = &format!("{}_window", pat_ident.ident);
                                    let window_arg_ident = syn::Pat::Ident(syn::PatIdent {
                                        attrs: Vec::new(),
                                        by_ref: None,
                                        mutability: None,
                                        ident: syn::Ident::new(window_arg_name, span),
                                        subpat: None,
                                    });
                                    let window_arg = syn::FnArg::Typed(syn::PatType {
                                        attrs: Vec::new(),
                                        pat: Box::new(window_arg_ident.clone()),
                                        colon_token: syn::token::Colon { spans: [span] },
                                        ty: Box::new(syn::Type::Verbatim(quote! { usize })),
                                    });

                                    let arg_local_name = &format!("{}_local", pat_ident.ident);
                                    let arg_local_ident = syn::Ident::new(arg_local_name, span);

                                    *arg = replaced_arg.clone();
                                    original_args[i] = replaced_arg;

                                    // See if we are mutable
                                    let new_slice = if is_mut {
                                        syn::parse(quote! {
                                            let #arg_local_ident: #arg_type = gpu::subslice_mut(#arg_name, __c * #window_arg_ident, #window_arg_ident);
                                        }.into()).expect("Failed to parse input as a statement 3")
                                    } else {
                                        syn::parse(quote! {
                                            let #arg_local_ident: #arg_type = gpu::subslice(#arg_name, __c * #window_arg_ident, #window_arg_ident);
                                        }.into()).expect("Failed to parse input as a statement 3")
                                    };
                                    stmts.push(new_slice);

                                    // Add
                                    wrapper_args.insert(i + 1, window_arg);

                                    // Add local to call args
                                    call_args_rev.push(arg_local_ident);
                                } else {
                                    panic!("Not a type in angle bracket");
                                }
                            } else {
                                panic!("No type for GpuChunkableMut");
                            }
                        } else {
                            // Add ident as is
                            let call_arg = syn::Ident::new(&format!("{}", pat_ident.ident), span);
                            call_args_rev.push(call_arg);
                        }
                    } else {
                        // Add ident as is
                        let call_arg = syn::Ident::new(&format!("{}", pat_ident.ident), span);
                        call_args_rev.push(call_arg);
                    }
                } else {
                    // Add ident as is
                    let call_arg = syn::Ident::new(&format!("{}", pat_ident.ident), span);
                    call_args_rev.push(call_arg);
                }
            } else {
                panic!("You don't have a name for your arg");
            }
        }
    }

    // 3. Call the actual function
    // Build call args:
    let func_ident = &func.sig.ident;
    let call_args: Vec<_> = call_args_rev.into_iter().rev().collect();
    // println!("{}", quote!(#func_ident(#(#call_args),*);));
    let call = syn::parse(
        quote! {
            #func_ident(#(#call_args),*);
        }
        .into(),
    )
    .expect("Failed to parse input as a statement 4");
    stmts.push(call);

    // Insert everything into the blocks of the wrapper
    wrapper_func.block.stmts.clear();
    wrapper_func.block.stmts.extend(stmts);

    wrapper_func
}

pub(crate) fn rewrite(_: TokenStream, input: TokenStream, use_gpu_codegen: bool) -> TokenStream {
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    let fun_span = fun.span();

    let mut wrapper_stream = proc_macro2::TokenStream::new();

    // The newly generated function uses the same span as the attributes
    let mut wrapper_fun = kernel_create_wrapper(&mut fun, fun_span);

    // Add proper device/kernel attributes to function signature and all closures inside the body.
    if use_gpu_codegen {
        crate::gpu_syntax::basic_rewrite_device(&mut fun);
        crate::gpu_syntax::basic_rewrite_kernel_entry(&mut wrapper_fun);
    }

    wrapper_fun.to_tokens(&mut wrapper_stream);

    fun.to_tokens(&mut wrapper_stream);

    // let source_code = wrapper_stream.to_string();
    // println!("{}", source_code);

    proc_macro::TokenStream::from(wrapper_stream)
}
