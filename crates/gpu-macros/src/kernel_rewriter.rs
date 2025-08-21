use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::spanned::Spanned;

fn kernel_rewrite_func(func: &mut syn::ItemFn, span: Span) {
    // Rewrite:
    // 1. Replace the function's attributes
    func.attrs.push(syn::parse_quote! {#[unsafe(no_mangle)]});

    let mut stmts = vec![];

    // 2. Add window to the argument list (also build the slices along the way)
    let func_args = &mut func.sig.inputs;

    // Backwards iteration so that insertion don't mess up the indices
    // Thanks the good idea of the LLM of Google Search
    for i in (0..func_args.len()).rev() {
        let arg = &mut func_args[i];
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
                                        ident: syn::Ident::new(
                                            &format!("__args_{}", pat_ident.ident),
                                            span,
                                        ),
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

                                    *arg = replaced_arg.clone();

                                    // See if we are mutable
                                    let new_slice = if is_mut {
                                        syn::parse(quote! {
                                            let #pat_ident: #arg_type = gpu::chunk_mut(#replaced_arg_name, #window_arg_ident, gpu::GpuChunkIdx::new());
                                        }.into()).expect("Failed to parse input as a statement 3")
                                    } else {
                                        syn::parse(quote! {
                                            let #pat_ident: #arg_type = gpu::chunk(#replaced_arg_name, #window_arg_ident, gpu::GpuChunkIdx::new());
                                        }.into()).expect("Failed to parse input as a statement 3")
                                    };
                                    stmts.push(new_slice);

                                    // Add
                                    func_args.insert(i + 1, window_arg);
                                } else {
                                    panic!("Not a type in angle bracket");
                                }
                            } else {
                                panic!("No type for GpuChunkableMut");
                            }
                        } else if path.path.segments[0].ident == "gpu"
                            && (path.path.segments[1].ident == "GpuChunkableMut2D"
                                || path.path.segments[1].ident == "GpuChunkable2D")
                        {
                            if let syn::PathArguments::AngleBracketed(slice_arg) =
                                &path.path.segments[1].arguments
                            {
                                if let syn::GenericArgument::Type(slice_ty) = &slice_arg.args[0] {
                                    let is_mut = path.path.segments[1].ident == "GpuChunkableMut2D";
                                    let replaced_arg_name = syn::Pat::Ident(syn::PatIdent {
                                        attrs: Vec::new(),
                                        by_ref: None,
                                        mutability: None,
                                        ident: syn::Ident::new(
                                            &format!("__args_{}", pat_ident.ident),
                                            span,
                                        ),
                                        subpat: None,
                                    });
                                    let arg_struct_name = syn::Pat::Ident(syn::PatIdent {
                                        attrs: Vec::new(),
                                        by_ref: None,
                                        mutability: None,
                                        ident: syn::Ident::new(
                                            &format!("__args_{}_struct", pat_ident.ident),
                                            span,
                                        ),
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
                                    let arg_struct_type = if is_mut {
                                        syn::Type::Verbatim(
                                            quote! { gpu::GpuChunkableMut2D<#slice_ty> },
                                        )
                                    } else {
                                        syn::Type::Verbatim(
                                            quote! { gpu::GpuChunkable2D<#slice_ty> },
                                        )
                                    };
                                    let size_x_arg_name =
                                        &format!("__args_{}_size_x", pat_ident.ident);
                                    let size_x_arg_ident = syn::Pat::Ident(syn::PatIdent {
                                        attrs: Vec::new(),
                                        by_ref: None,
                                        mutability: None,
                                        ident: syn::Ident::new(size_x_arg_name, span),
                                        subpat: None,
                                    });
                                    let size_x_arg = syn::FnArg::Typed(syn::PatType {
                                        attrs: Vec::new(),
                                        pat: Box::new(size_x_arg_ident.clone()),
                                        colon_token: syn::token::Colon { spans: [span] },
                                        ty: Box::new(syn::Type::Verbatim(quote! { usize })),
                                    });

                                    // See if we are mutable
                                    let new_struct = if is_mut {
                                        syn::parse(quote! {
                                            let mut #arg_struct_name: #arg_struct_type = unsafe{gpu::GpuChunkableMut2D::new_from_gpu(#replaced_arg_name, #size_x_arg_ident)};
                                        }.into()).expect("Failed to parse input as a statement 4")
                                    } else {
                                        syn::parse(quote! {
                                            let #arg_struct_name: #arg_struct_type = unsafe{ gpu::GpuChunkable2D::new_from_gpu(
                                                #replaced_arg_name,
                                                #size_x_arg_ident,
                                            )};
                                        }.into()).expect("Failed to parse input as a statement 3")
                                    };
                                    stmts.push(new_struct);

                                    let new_struct_ref = if is_mut {
                                        syn::parse(
                                            quote! {
                                                let #pat_ident = &mut #arg_struct_name;
                                            }
                                            .into(),
                                        )
                                        .expect("Failed to parse input as a statement 5")
                                    } else {
                                        syn::parse(
                                            quote! {
                                                let #pat_ident = &#arg_struct_name;
                                            }
                                            .into(),
                                        )
                                        .expect("Failed to parse input as a statement 6")
                                    };
                                    stmts.push(new_struct_ref);

                                    *arg = replaced_arg.clone();

                                    // Add
                                    func_args.insert(i + 1, size_x_arg);
                                } else {
                                    panic!("Not a type in angle bracket");
                                }
                            } else {
                                panic!("No type for GpuChunkableMut2D");
                            }
                        }
                    }
                }
            } else {
                panic!("You don't have a name for your arg");
            }
        }
    }

    // Insert everything into the front of the func's block
    func.block.stmts.splice(0..0, stmts.iter().cloned());
}

pub(crate) fn rewrite(
    _: TokenStream,
    input: TokenStream,
    target: crate::CodegenTarget,
) -> TokenStream {
    let mut fun = syn::parse_macro_input!(input as syn::ItemFn);
    let fun_span = fun.span();

    let mut new_stream = proc_macro2::TokenStream::new();

    kernel_rewrite_func(&mut fun, fun_span);

    // Add proper device/kernel attributes to function signature and all closures inside the body.
    crate::gpu_syntax::basic_rewrite_gpu_func(&mut fun, true, target);

    fun.to_tokens(&mut new_stream);

    // let source_code = new_stream.to_string();
    // println!("{}", new_stream);

    proc_macro::TokenStream::from(new_stream)
}
