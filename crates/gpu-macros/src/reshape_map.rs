use proc_macro::TokenStream;
use quote::quote_spanned;
use syn::parse::Parse;
use syn::spanned::Spanned;
use syn::{Expr, ExprArray, ExprBinary, ExprGroup, ExprTuple, ExprUnary, Ident, Path};

fn check_permutation(values: &[usize]) -> Result<(), syn::Error> {
    let n = values.len();

    // Check permutation validity
    let mut seen = vec![false; n];
    for &v in values {
        if v >= n {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("value {} out of range (0..{})", v, n),
            ));
        }
        if seen[v] {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("duplicated idx {}", v),
            ));
        }
        seen[v] = true;
    }
    Ok(())
}

fn get_size_pairs(exprs: impl Iterator<Item = Expr>) -> Result<(Vec<Expr>, Vec<Expr>), syn::Error> {
    let mut old_sizes = vec![];
    let mut new_sizes = vec![];
    for e in exprs {
        match e {
            Expr::Tuple(ExprTuple { elems, .. }) if elems.len() == 2 => {
                old_sizes.push(elems[0].clone());
                new_sizes.push(elems[1].clone());
            }
            Expr::Tuple(_) => {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "size pair must have exactly two elements",
                ));
            }
            _ => {
                old_sizes.push(e.clone());
                new_sizes.push(e);
            }
        }
    }
    Ok((old_sizes, new_sizes))
}

// map!([4,4] | [2,2] => layout: [0, 1, 2, 3], offset: 10)
struct MapPermuteArgs {
    span: proc_macro2::Span,
    gpu_crate: Path,
    idx_sizes: ExprArray,
    _semi: syn::Token![|],
    tid_sizes: ExprArray,
    layout: Option<ExprArray>,
    offset: Option<Expr>,
}

impl Parse for MapPermuteArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let span = input.span();
        let gpu_crate = input.parse()?;
        let _comma: syn::Token![,] = input.parse()?;
        let idx_sizes = input.parse()?;
        let _semi: syn::Token![|] = input.parse()?;
        let tid_sizes = input.parse()?;
        let mut layout = None;
        let mut offset = None;
        if input.peek(syn::Token![=>]) {
            let _: syn::Token![=>] = input.parse()?;
            loop {
                let kind: Ident = input.parse()?;
                let _: syn::Token![:] = input.parse()?;
                match kind.to_string().as_str() {
                    "layout" => {
                        let expr = input.parse()?;
                        layout = Some(expr);
                    }
                    "offset" => {
                        let expr = input.parse()?;
                        offset = Some(expr);
                    }
                    _ => {
                        return Err(syn::Error::new(
                            kind.span(),
                            "Expected optional 'layout' and 'offset'",
                        ));
                    }
                }
                if input.peek(syn::Token![,]) {
                    let _: syn::Token![,] = input.parse()?;
                    continue;
                } else {
                    break;
                }
            }
        }
        Ok(MapPermuteArgs { gpu_crate, span, idx_sizes, _semi, tid_sizes, layout, offset })
    }
}

pub(crate) fn map_reshape_params(tokens: TokenStream) -> TokenStream {
    let args: MapPermuteArgs = syn::parse(tokens).expect(
        "Map args must be of the form map_reshape_params!([sizes] | [extra_sizes] => [...])",
    );
    let local_id_len = args.idx_sizes.elems.len();
    let tid_len = args.tid_sizes.elems.len();
    let len = local_id_len + tid_len;
    let mut all_sizes = args.idx_sizes.clone();
    if tid_len < 1 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "thread index must have at least one dimension",
        )
        .to_compile_error()
        .into();
    }

    if local_id_len < 1 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "local index must have at least one dimension",
        )
        .to_compile_error()
        .into();
    }

    all_sizes.elems.extend(args.tid_sizes.elems.iter().cloned());
    let span = args.span;
    let (permutation, reversed) = if let Some(layout) = args.layout {
        match get_const_array(layout.elems.iter().cloned(), local_id_len) {
            Ok(l) => l,
            Err(e) => return e.to_compile_error().into(),
        }
    } else {
        ((0..len).collect::<Vec<_>>(), vec![false; len])
    };
    if let Err(e) = check_permutation(&permutation) {
        return e.to_compile_error().into();
    }
    let (old_sizes, new_sizes) = match get_size_pairs(all_sizes.elems.iter().cloned()) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    let offset = if let Some(offset) = args.offset {
        quote_spanned! {
            span => #offset
        }
    } else {
        quote_spanned! {
            span => 0
        }
    };

    let mut cached_weights = vec![];
    for i in 0..len {
        cached_weights.push(all_sizes.elems[i].clone());
    }
    for i in 0..len {
        let span = all_sizes.elems[i].span();
        let mut w = None;
        for &proj in permutation.iter().take(i) {
            let s = new_sizes[proj].clone();
            w = if let Some(oldw) = w {
                Some(Expr::Binary(ExprBinary {
                    attrs: vec![],
                    left: Box::new(oldw),
                    right: Box::new(s.clone()),
                    op: syn::BinOp::Mul(syn::token::Star { spans: [span] }),
                }))
            } else {
                Some(s.clone())
            }
        }
        cached_weights[permutation[i]] = w.unwrap_or_else(|| syn::parse_quote! {1});
    }

    let lid = Ident::new("lid", span);
    let tid = Ident::new("tid", span);
    let global_idx = Ident::new("global_idx", span);
    let valid = Ident::new("valid", span);
    let remain_tid = Ident::new("remain_tid", span);
    let weights = Ident::new("weights", span);
    let self_ident = Ident::new("self", span);
    let mut gen_global_id = quote_spanned! {span =>
        let mut #global_idx = #self_ident.offset;
        let mut #valid = true;
        let #remain_tid = #tid;
    };

    for (i, &is_reversed) in reversed.iter().enumerate() {
        let span = all_sizes.elems[i].span();
        let old_s_i = Ident::new(&format!("old_s_{}", i), span);
        let new_s_i = Ident::new(&format!("new_s_{}", i), span);
        let current_lid = if local_id_len == 1 {
            quote_spanned! { span => #lid }
        } else {
            let i = syn::Member::from(i);
            quote_spanned! { span => #lid.#i }
        };
        let current_id = if i < local_id_len {
            quote_spanned! { span => #current_lid  }
        } else if i < len - 1 {
            quote_spanned! { span => (#remain_tid % #old_s_i) }
        } else {
            quote_spanned! { span => (#remain_tid) }
        };
        let id_i = if is_reversed {
            Expr::Verbatim(quote_spanned! { span =>
                (#new_s_i - 1 - #current_id)
            })
        } else {
            Expr::Verbatim(quote_spanned! { span =>
                #current_id
            })
        };

        // If the size is an expression that is not dynamic to the function,
        // we can directly use it instead of loading from an array.
        let get_non_dynamic_expr = |e: &Expr| match e {
            Expr::Lit(_) => Some(e.clone()),
            Expr::Group(ExprGroup { expr, .. }) => match &**expr {
                Expr::Lit(_) => Some(e.clone()),
                _ => None,
            },
            Expr::Call(call_expr) if call_expr.args.is_empty() => Some(e.clone()),
            Expr::Const(_) => Some(e.clone()),
            _ => None,
        };

        let old_size = get_non_dynamic_expr(&old_sizes[i]);
        let new_size = get_non_dynamic_expr(&new_sizes[i]);
        gen_global_id = new_size.map_or(
            quote_spanned! { span =>
                #gen_global_id
                let #new_s_i = #self_ident.new_sizes[#i];
            },
            |new_size| {
                quote_spanned! { span =>
                    #gen_global_id
                    let #new_s_i = #new_size;
                }
            },
        );

        gen_global_id = old_size.map_or(
            quote_spanned! { span =>
                #gen_global_id
                let #old_s_i = #self_ident.old_sizes[#i];
            },
            |old_size| {
                quote_spanned! { span =>
                    #gen_global_id
                    let #old_s_i = #old_size;
                }
            },
        );

        gen_global_id = quote_spanned! { span =>
            #gen_global_id
            #global_idx += #id_i * #weights[#i];
            #valid &= (#id_i < #new_s_i) && (#id_i < #old_s_i);
        };
        // avoid warning: value assigned to `remain_id` is never read
        if i >= local_id_len {
            gen_global_id = quote_spanned! { span =>
                #gen_global_id
                let #remain_tid = #remain_tid / #old_s_i;
            };
        }
    }
    // ensure all tid are consumed
    gen_global_id = quote_spanned! { span =>
        #gen_global_id
        #valid &= #remain_tid == 0;
    };

    // === Companion: gen_lid_offset computes ONLY the lid-portion of the index.
    // This is the same per-dim arithmetic as above, but restricted to i < local_id_len
    // (skipping tid dims). Used by OpenedTile to cache a pre-offset 64-bit pointer
    // so per-access only pays compile-time-const offset arithmetic.
    let lid_offset_ident = Ident::new("lid_offset", span);
    let lid_valid_ident = Ident::new("lid_valid", span);
    let mut gen_lid_offset = quote_spanned! {span =>
        let mut #lid_offset_ident: u32 = 0;
        let mut #lid_valid_ident: bool = true;
    };
    for (i, &is_reversed) in reversed.iter().enumerate().take(local_id_len) {
        let span = all_sizes.elems[i].span();
        let old_s_i = Ident::new(&format!("lo_old_s_{}", i), span);
        let new_s_i = Ident::new(&format!("lo_new_s_{}", i), span);
        let current_lid = if local_id_len == 1 {
            quote_spanned! { span => #lid }
        } else {
            let mi = syn::Member::from(i);
            quote_spanned! { span => #lid.#mi }
        };
        let id_i = if is_reversed {
            Expr::Verbatim(quote_spanned! { span =>
                (#new_s_i - 1 - #current_lid)
            })
        } else {
            Expr::Verbatim(quote_spanned! { span =>
                #current_lid
            })
        };
        let get_non_dynamic_expr = |e: &Expr| match e {
            Expr::Lit(_) => Some(e.clone()),
            Expr::Group(ExprGroup { expr, .. }) => match &**expr {
                Expr::Lit(_) => Some(e.clone()),
                _ => None,
            },
            Expr::Call(call_expr) if call_expr.args.is_empty() => Some(e.clone()),
            Expr::Const(_) => Some(e.clone()),
            _ => None,
        };
        let old_size = get_non_dynamic_expr(&old_sizes[i]);
        let new_size = get_non_dynamic_expr(&new_sizes[i]);
        gen_lid_offset = new_size.map_or(
            quote_spanned! { span =>
                #gen_lid_offset
                let #new_s_i = #self_ident.new_sizes[#i];
            },
            |new_size| {
                quote_spanned! { span =>
                    #gen_lid_offset
                    let #new_s_i = #new_size;
                }
            },
        );
        gen_lid_offset = old_size.map_or(
            quote_spanned! { span =>
                #gen_lid_offset
                let #old_s_i = #self_ident.old_sizes[#i];
            },
            |old_size| {
                quote_spanned! { span =>
                    #gen_lid_offset
                    let #old_s_i = #old_size;
                }
            },
        );
        gen_lid_offset = quote_spanned! { span =>
            #gen_lid_offset
            #lid_offset_ident += #id_i * #weights[#i];
            #lid_valid_ident &= (#id_i < #new_s_i) && (#id_i < #old_s_i);
        };
    }

    // === Companion: gen_thread_base computes `struct_offset + sum(tid-dim contributions)`.
    // By construction: `map(lid, tid).1 == thread_base(tid).1 + map_lid_offset(lid).1`.
    let tb_offset_ident = Ident::new("tb_offset", span);
    let tb_valid_ident = Ident::new("tb_valid", span);
    let tb_remain_ident = Ident::new("tb_remain", span);
    let tb_tid_ident = Ident::new("tb_tid", span);
    let mut gen_thread_base = quote_spanned! {span =>
        let mut #tb_offset_ident: u32 = #self_ident.offset;
        let mut #tb_valid_ident: bool = true;
        let #tb_remain_ident = #tb_tid_ident;
    };
    for (i, &is_reversed) in reversed.iter().enumerate() {
        if i < local_id_len {
            continue;
        }
        let span = all_sizes.elems[i].span();
        let old_s_i = Ident::new(&format!("tb_old_s_{}", i), span);
        let new_s_i = Ident::new(&format!("tb_new_s_{}", i), span);
        let current_id = if i < len - 1 {
            quote_spanned! { span => (#tb_remain_ident % #old_s_i) }
        } else {
            quote_spanned! { span => (#tb_remain_ident) }
        };
        let id_i = if is_reversed {
            Expr::Verbatim(quote_spanned! { span =>
                (#new_s_i - 1 - #current_id)
            })
        } else {
            Expr::Verbatim(quote_spanned! { span =>
                #current_id
            })
        };
        let get_non_dynamic_expr = |e: &Expr| match e {
            Expr::Lit(_) => Some(e.clone()),
            Expr::Group(ExprGroup { expr, .. }) => match &**expr {
                Expr::Lit(_) => Some(e.clone()),
                _ => None,
            },
            Expr::Call(call_expr) if call_expr.args.is_empty() => Some(e.clone()),
            Expr::Const(_) => Some(e.clone()),
            _ => None,
        };
        let old_size = get_non_dynamic_expr(&old_sizes[i]);
        let new_size = get_non_dynamic_expr(&new_sizes[i]);
        gen_thread_base = new_size.map_or(
            quote_spanned! { span =>
                #gen_thread_base
                let #new_s_i = #self_ident.new_sizes[#i];
            },
            |new_size| {
                quote_spanned! { span =>
                    #gen_thread_base
                    let #new_s_i = #new_size;
                }
            },
        );
        gen_thread_base = old_size.map_or(
            quote_spanned! { span =>
                #gen_thread_base
                let #old_s_i = #self_ident.old_sizes[#i];
            },
            |old_size| {
                quote_spanned! { span =>
                    #gen_thread_base
                    let #old_s_i = #old_size;
                }
            },
        );
        gen_thread_base = quote_spanned! { span =>
            #gen_thread_base
            #tb_offset_ident += #id_i * #weights[#i];
            #tb_valid_ident &= (#id_i < #new_s_i) && (#id_i < #old_s_i);
            let #tb_remain_ident = #tb_remain_ident / #old_s_i;
        };
    }
    gen_thread_base = quote_spanned! { span =>
        #gen_thread_base
        #tb_valid_ident &= #tb_remain_ident == 0;
    };

    let gpu_crate = args.gpu_crate;
    let unit_index_ty = quote_spanned! { span => u32 };
    let index_type = if local_id_len == 1 {
        unit_index_ty
    } else {
        let tys = vec![quote_spanned! { span => u32 }; local_id_len];
        quote_spanned! { span => (#(#tys),*) }
    };
    let gen_map_with_rows_impl = if local_id_len == 2 {
        let row_ident = Ident::new("row", span);
        let col_ident = Ident::new("col", span);
        let row_offset_ident = Ident::new("row_offset", span);
        let row_valid_ident = Ident::new("row_valid", span);
        let col_offset_ident = Ident::new("col_offset", span);
        let col_valid_ident = Ident::new("col_valid", span);
        let gen_lid_split_offset =
            |i: usize, lid_ident: &Ident, offset_ident: &Ident, valid_ident: &Ident| {
                let span = all_sizes.elems[i].span();
                let old_s_i = Ident::new(&format!("split_old_s_{}", i), span);
                let new_s_i = Ident::new(&format!("split_new_s_{}", i), span);
                let id_i = if reversed[i] {
                    Expr::Verbatim(quote_spanned! { span =>
                        (#new_s_i - 1 - #lid_ident)
                    })
                } else {
                    Expr::Verbatim(quote_spanned! { span =>
                        #lid_ident
                    })
                };
                let get_non_dynamic_expr = |e: &Expr| match e {
                    Expr::Lit(_) => Some(e.clone()),
                    Expr::Group(ExprGroup { expr, .. }) => match &**expr {
                        Expr::Lit(_) => Some(e.clone()),
                        _ => None,
                    },
                    Expr::Call(call_expr) if call_expr.args.is_empty() => Some(e.clone()),
                    Expr::Const(_) => Some(e.clone()),
                    _ => None,
                };
                let old_size = get_non_dynamic_expr(&old_sizes[i]);
                let new_size = get_non_dynamic_expr(&new_sizes[i]);
                let mut gen_split_offset = quote_spanned! {span =>
                    let mut #offset_ident: u32 = 0;
                    let mut #valid_ident: bool = true;
                };
                gen_split_offset = new_size.map_or(
                    quote_spanned! { span =>
                        #gen_split_offset
                        let #new_s_i = #self_ident.new_sizes[#i];
                    },
                    |new_size| {
                        quote_spanned! { span =>
                            #gen_split_offset
                            let #new_s_i = #new_size;
                        }
                    },
                );
                gen_split_offset = old_size.map_or(
                    quote_spanned! { span =>
                        #gen_split_offset
                        let #old_s_i = #self_ident.old_sizes[#i];
                    },
                    |old_size| {
                        quote_spanned! { span =>
                            #gen_split_offset
                            let #old_s_i = #old_size;
                        }
                    },
                );
                quote_spanned! { span =>
                    #gen_split_offset
                    #offset_ident += #id_i * #weights[#i];
                    #valid_ident &= (#id_i < #new_s_i) && (#id_i < #old_s_i);
                }
            };
        let gen_row_offset =
            gen_lid_split_offset(1, &row_ident, &row_offset_ident, &row_valid_ident);
        let gen_in_row_offset =
            gen_lid_split_offset(0, &col_ident, &col_offset_ident, &col_valid_ident);
        quote_spanned! { span =>
            unsafe impl<CS: #gpu_crate::chunk_scope::ChunkScope> #gpu_crate::chunk::MapWithRows<CS>
                for PrivateMapReshapeShuffle
            {
                fn row_lid_offset(&#self_ident, #row_ident: u32) -> (bool, u32) {
                    let #weights = #self_ident.cached_weights;
                    #gen_row_offset
                    (#row_valid_ident, #row_offset_ident)
                }

                fn in_row_lid_offset(&#self_ident, #col_ident: u32) -> (bool, u32) {
                    let #weights = #self_ident.cached_weights;
                    #gen_in_row_offset
                    (#col_valid_ident, #col_offset_ident)
                }
            }
        }
    } else {
        quote_spanned! { span => }
    };
    quote_spanned! { span => {
        #[derive(Clone, Copy)]
        #[allow(dead_code)]
        struct PrivateMapReshapeShuffle{
            old_sizes: [u32; #len],
            new_sizes: [u32; #len],
            offset: u32,
            cached_weights: [u32; #len],
        }
        unsafe impl<CS: #gpu_crate::chunk_scope::ChunkScope> #gpu_crate::chunk::ScopeUniqueMap<CS> for PrivateMapReshapeShuffle
        {
            type IndexType = #index_type;
            type GlobalIndexType = u32;

            fn map(&#self_ident, #lid: Self::IndexType, thread_ids: [u32; #gpu_crate::chunk_scope::TID_MAX_LEN]) -> (bool, Self::GlobalIndexType) {
                 let #tid = CS::global_id_x(thread_ids)
            + CS::global_dim_x()
                * (CS::global_id_y(thread_ids) + CS::global_dim_y() * CS::global_id_z(thread_ids));
                assert!(#tid %  CS::global_dim_x() == CS::global_id_x(thread_ids));
                assert!(#tid /  CS::global_dim_x() == CS::global_id_y(thread_ids) + CS::global_dim_y() * CS::global_id_z(thread_ids));
                assert!((#tid /  CS::global_dim_x()) % CS::global_dim_y() == CS::global_id_y(thread_ids));
                assert!((#tid /  CS::global_dim_x()) / CS::global_dim_y() == CS::global_id_z(thread_ids));
            let #weights = #self_ident.cached_weights;
            #gen_global_id
            (#valid, #global_idx)
        }
    }
    unsafe impl<CS: #gpu_crate::chunk_scope::ChunkScope> #gpu_crate::chunk::MapWithLidOffset<CS> for PrivateMapReshapeShuffle {
        // Companion to `map`: returns ONLY the lid-portion of the linear offset
        // (no struct offset, no thread contribution). Used by `open_tile` to let
        // unrolled tile accesses reduce to `tile_ptr + const_offset` without
        // re-materializing the thread base on every access.
        //
        // Safety invariant (per trait contract):
        //   map(lid, tid).1 == map(<lid=0>, tid).1 + map_lid_offset(lid).1
        fn map_lid_offset(&#self_ident, #lid: #index_type) -> (bool, u32) {
            let #weights = #self_ident.cached_weights;
            #gen_lid_offset
            (#lid_valid_ident, #lid_offset_ident)
        }

        fn thread_base(&#self_ident, thread_ids: [u32; #gpu_crate::chunk_scope::TID_MAX_LEN]) -> (bool, u32) {
            let #tb_tid_ident = CS::global_id_x(thread_ids)
                + CS::global_dim_x()
                    * (CS::global_id_y(thread_ids) + CS::global_dim_y() * CS::global_id_z(thread_ids));
            let #weights = #self_ident.cached_weights;
            #gen_thread_base
            (#tb_valid_ident, #tb_offset_ident)
        }
    }
    #gen_map_with_rows_impl
    #[allow(clippy::identity_op)]
    PrivateMapReshapeShuffle {
        old_sizes: [#(#old_sizes,)*],
        new_sizes:  [#(#new_sizes,)*],
        offset:     #offset,
        cached_weights: [#(#cached_weights,)*]
    }
    }}.into()
}

fn get_const_array(
    exprs: impl Iterator<Item = Expr>,
    lid_len: usize,
) -> Result<(Vec<usize>, Vec<bool>), syn::Error> {
    let mut const_array = vec![];
    let mut const_neg = vec![];

    let get_const_val = |size: Expr| match size {
        Expr::Lit(expr_lit) => {
            if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                let val = lit_int.base10_parse::<usize>().unwrap();
                Some(val)
            } else {
                None
            }
        }
        Expr::Group(ExprGroup { expr, .. }) => match &*expr {
            Expr::Lit(expr_lit) => {
                if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                    let val = lit_int.base10_parse::<usize>().unwrap();
                    Some(val)
                } else {
                    None
                }
            }
            _ => None,
        },
        Expr::Path(path) => {
            // use i0..iN represent 0..N, t0..tM represent N+0..N+M
            if path.path.segments.len() == 1 {
                let ident = &path.path.segments[0].ident;
                let s = ident.to_string();
                let is_local_id = s.starts_with('i');
                let is_thread_id = s.starts_with('t');
                let offset = if is_thread_id { lid_len } else { 0 };
                if is_local_id || is_thread_id {
                    if let Ok(v) = s[1..].parse::<usize>() { Some(v + offset) } else { None }
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    };
    for expr in exprs {
        let span = expr.span();
        let (val, neg) = match expr {
            Expr::Unary(ExprUnary { op: syn::UnOp::Neg(_), expr, .. }) => {
                (get_const_val(*expr), true)
            }
            _ => (get_const_val(expr), false),
        };
        let Some(val) = val else {
            return Err(syn::Error::new(
                span,
                "All elements in layout must be constant usize or i0..iN/t0..tM",
            ));
        };
        const_array.push(val);
        const_neg.push(neg);
    }
    let mut reversed = vec![false; const_neg.len()];
    for (i, &val) in const_array.iter().enumerate() {
        reversed[val] = const_neg[i];
    }
    Ok((const_array, reversed))
}
