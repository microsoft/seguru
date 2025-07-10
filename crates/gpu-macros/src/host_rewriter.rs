use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::spanned::Spanned;

fn host_create_wrapper(func: &syn::ItemFn, span: Span) -> syn::ItemFn {
    // Rewrite:
    // 1. Create a new function and change its name and attributes
    let mut wrapper_func = func.clone();
    wrapper_func.sig.ident = syn::Ident::new(&format!("launch_{}", &func.sig.ident), span);
    let kernel_func_str = &format!("{}_wrapper", &func.sig.ident);

    wrapper_func.attrs.clear();
    wrapper_func.sig.output = syn::ReturnType::Type(
        syn::token::RArrow::default(),       // ->
        Box::new(syn::parse_quote! { i32 }), // u32
    );
    // Allows name mangling since it's used locally
    // wrapper_func.attrs.push(syn::parse_quote! {#[unsafe(no_mangle)]});

    let mut stmts = vec![];

    // 2. Add window to the argument list (also build the ptr and args along the way)
    let wrapper_args = &wrapper_func.sig.inputs;
    let mut call_args_rev = vec![];

    // Backwards iteration so that insertion don't mess up the indices
    // Thanks the good idea of the LLM of Google Search
    for i in (0..wrapper_args.len()).rev() {
        let arg = &wrapper_args[i];
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                if let syn::Type::Reference(type_ref) = &*pat_type.ty {
                    if let syn::Type::Path(path) = &*type_ref.elem {
                        // It MUST follow the gpu::GpuChunkable(Mut)
                        if path.path.segments[0].ident == "gpu"
                            && (path.path.segments[1].ident == "GpuChunkableMut"
                                || path.path.segments[1].ident == "GpuChunkable")
                        {
                            let arg_ptr_ident =
                                syn::Ident::new(&format!("{}_ptr", pat_ident.ident), span);
                            let arg_ptr_rptr_ident =
                                syn::Ident::new(&format!("{}_ptr_rptr", pat_ident.ident), span);
                            let arg_ptr_ptr_ident =
                                syn::Ident::new(&format!("{}_ptr_ptr", pat_ident.ident), span);

                            let arg_size_ident =
                                syn::Ident::new(&format!("{}_size", pat_ident.ident), span);
                            let arg_size_rptr_ident =
                                syn::Ident::new(&format!("{}_size_rptr", pat_ident.ident), span);
                            let arg_size_ptr_ident =
                                syn::Ident::new(&format!("{}_size_ptr", pat_ident.ident), span);

                            let arg_window_rptr_ident = syn::Ident::new(
                                &format!("{}_window_ptr_rptr", pat_ident.ident),
                                span,
                            );
                            let arg_window_ptr_ident = syn::Ident::new(
                                &format!("{}_window_ptr_ptr", pat_ident.ident),
                                span,
                            );

                            // Build up the ptrs needed in args
                            let arg_stub_vec = vec![
                                syn::parse(quote! {
                                        let #arg_ptr_ident = #pat_ident.slice.as_ptr() as *const ::std::os::raw::c_void;
                                    }.into()).expect("Failed to parse input as a statement 8"),
                                syn::parse(quote! {
                                        let #arg_ptr_rptr_ident: *const *const ::std::os::raw::c_void = &#arg_ptr_ident;
                                    }.into()).expect("Failed to parse input as a statement 9"),
                                syn::parse(quote! {
                                        let #arg_ptr_ptr_ident = #arg_ptr_rptr_ident as *const ::std::os::raw::c_void;
                                    }.into()).expect("Failed to parse input as a statement 10"),
                                syn::parse(quote! {
                                        let #arg_size_ident = #pat_ident.slice.len();
                                    }.into()).expect("Failed to parse input as a statement 11"),
                                syn::parse(quote! {
                                        let #arg_size_rptr_ident: *const usize = &#arg_size_ident;
                                    }.into()).expect("Failed to parse input as a statement 12"),
                                syn::parse(quote! {
                                        let #arg_size_ptr_ident: *const ::std::os::raw::c_void = #arg_size_rptr_ident as *const ::std::os::raw::c_void;
                                    }.into()).expect("Failed to parse input as a statement 13"),
                                syn::parse(quote! {
                                        let #arg_window_rptr_ident: *const usize = &#pat_ident.window;
                                    }.into()).expect("Failed to parse input as a statement 14"),
                                syn::parse(quote! {
                                        let #arg_window_ptr_ident: *const ::std::os::raw::c_void =
                                            #arg_window_rptr_ident as *const ::std::os::raw::c_void;
                                    }.into()).expect("Failed to parse input as a statement 15"),
                            ];
                            for arg_stub_stmt in arg_stub_vec {
                                stmts.push(arg_stub_stmt);
                            }

                            // Add local to call args in a reversed manner
                            call_args_rev.push(arg_window_ptr_ident);
                            call_args_rev.push(arg_size_ptr_ident.clone());
                            call_args_rev.push(arg_ptr_ptr_ident.clone());
                        } else {
                            // Add ident as is
                            let call_arg = syn::Ident::new(&format!("{}", pat_ident.ident), span);
                            call_args_rev.push(call_arg);
                        }
                    } else if let syn::Type::Slice(_) = &*type_ref.elem {
                        // Slices requires two components: ptr and size
                        let arg_ptr_ident =
                            syn::Ident::new(&format!("{}_ptr", pat_ident.ident), span);
                        let arg_ptr_rptr_ident =
                            syn::Ident::new(&format!("{}_ptr_rptr", pat_ident.ident), span);
                        let arg_ptr_ptr_ident =
                            syn::Ident::new(&format!("{}_ptr_ptr", pat_ident.ident), span);

                        let arg_size_ident =
                            syn::Ident::new(&format!("{}_size", pat_ident.ident), span);
                        let arg_size_rptr_ident =
                            syn::Ident::new(&format!("{}_size_rptr", pat_ident.ident), span);
                        let arg_size_ptr_ident =
                            syn::Ident::new(&format!("{}_size_ptr", pat_ident.ident), span);

                        let arg_stub_vec = vec![
                            syn::parse(quote! {
                                    let #arg_ptr_ident = #pat_ident.as_ptr() as *const ::std::os::raw::c_void;
                                }.into()).expect("Failed to parse input as a statement 8"),
                            syn::parse(quote! {
                                    let #arg_ptr_rptr_ident: *const *const ::std::os::raw::c_void = &#arg_ptr_ident;
                                }.into()).expect("Failed to parse input as a statement 9"),
                            syn::parse(quote! {
                                    let #arg_ptr_ptr_ident = #arg_ptr_rptr_ident as *const ::std::os::raw::c_void;
                                }.into()).expect("Failed to parse input as a statement 10"),
                            syn::parse(quote! {
                                    let #arg_size_ident = #pat_ident.len();
                                }.into()).expect("Failed to parse input as a statement 11"),
                            syn::parse(quote! {
                                    let #arg_size_rptr_ident: *const usize = &#arg_size_ident;
                                }.into()).expect("Failed to parse input as a statement 12"),
                            syn::parse(quote! {
                                    let #arg_size_ptr_ident: *const ::std::os::raw::c_void = #arg_size_rptr_ident as *const ::std::os::raw::c_void;
                                }.into()).expect("Failed to parse input as a statement 13"),
                        ];
                        for arg_stub_stmt in arg_stub_vec {
                            stmts.push(arg_stub_stmt);
                        }

                        // Add local to call args in a reversed manner
                        call_args_rev.push(arg_size_ptr_ident.clone());
                        call_args_rev.push(arg_ptr_ptr_ident.clone());
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

    // 3. Insert the config argument

    // 3. Call the actual function
    // Build call args:
    let call_args: Vec<_> = call_args_rev.into_iter().rev().collect();
    let shared_mem_size_ident =
        syn::Ident::new(&format!("const_share_size_{}", &func.sig.ident), span);
    // println!("{}", quote!(#func_ident(#(#call_args),*);));
    let call_stub_vec = vec![
        syn::parse(quote! {
            let args_for_launching: &mut [*const ::std::os::raw::c_void] = &mut [
                #(#call_args),*
            ];
        }.into()).expect("Failed to parse input as a statement 16"),
        syn::parse(quote! {
            let func_name_cstr = std::ffi::CString::new(#kernel_func_str).unwrap();
        }.into()).expect("Failed to parse input as a statement 17"),
        syn::parse(quote! {
            let func_name = func_name_cstr.as_ptr() as *const ::std::os::raw::c_char;
        }.into()).expect("Failed to parse input as a statement 18"),
        syn::parse(quote! {
            let args_for_launching_ptr = args_for_launching.as_mut_ptr() as *mut ::std::os::raw::c_void;
        }.into()).expect("Failed to parse input as a statement 19"),
        syn::parse(quote! {
            let mut res;
        }.into()).expect("Failed to parse input as a statement 20"),
        syn::parse(quote! {
            unsafe {
                res = cuda_bindings::gpu_launch_kernel(
                    func_name,
                    config.grid_dim_x,
                    config.grid_dim_y,
                    config.grid_dim_z,
                    config.block_dim_x,
                    config.block_dim_y,
                    config.block_dim_z,
                    #shared_mem_size_ident as u32,
                    args_for_launching_ptr,
                    core::ptr::null_mut(),
                );
            }
        }.into()).expect("Failed to parse input as a statement 21"),
        syn::parse(quote! {
            if res != 0 {
                return res;
            }
        }.into()).expect("Failed to parse input as a statement 22"),
        syn::parse(quote! {
            unsafe {
                res = cuda_bindings::gpu_device_sync();
            }
        }.into()).expect("Failed to parse input as a statement 23"),
        syn::parse(quote! {
            return res;
        }.into()).expect("Failed to parse input as a statement 24"),
    ];
    for call_stub_stmt in call_stub_vec {
        stmts.push(call_stub_stmt);
    }

    // Insert everything into the blocks of the wrapper
    // wrapper_func.block.stmts.clear();
    wrapper_func.block.stmts.extend(stmts);

    wrapper_func
}

pub(crate) fn rewrite(attr: TokenStream, input: TokenStream) -> TokenStream {
    let fun = syn::parse_macro_input!(input as syn::ItemFn);
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
    let wrapper_fun = host_create_wrapper(&fun, fun_span);
    wrapper_fun.to_tokens(&mut wrapper_stream);

    let wrapper_stream_macro: proc_macro::TokenStream = wrapper_stream.into();
    const_val_def.extend(wrapper_stream_macro);

    // Original function is dropped and replaced with our stuff

    // let source_code = const_val_def.to_string();
    // println!("{}", source_code);

    const_val_def
}
