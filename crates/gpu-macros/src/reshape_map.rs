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
    let remain_lid = Ident::new("remain_id", span);
    let remain_tid = Ident::new("remain_tid", span);
    let old_s = Ident::new("old_s", span);
    let new_s = Ident::new("new_s", span);
    let weights = Ident::new("weights", span);
    let mut gen_global_id = quote_spanned! {span =>
        let mut #global_idx = self.offset;
        let mut #valid = true;
        let #remain_lid = #lid;
        let #remain_tid = #tid;
    };

    for (i, &is_reversed) in reversed.iter().enumerate() {
        let span = all_sizes.elems[i].span();
        let old_s_i = Ident::new(&format!("old_s_{}", i), span);
        let new_s_i = Ident::new(&format!("new_s_{}", i), span);
        let remain_id = if i < local_id_len { remain_lid.clone() } else { remain_tid.clone() };

        let id_i = if is_reversed {
            Expr::Verbatim(quote_spanned! { span =>
                (#new_s_i - 1 - (#remain_id % #old_s_i))
            })
        } else {
            Expr::Verbatim(quote_spanned! { span =>
                (#remain_id % #old_s_i)
            })
        };
        gen_global_id = if i < local_id_len {
            quote_spanned! { span =>
                #gen_global_id
                let #new_s_i = #new_s[#i];
                let #old_s_i = #old_s[#i];
                #global_idx += #id_i * #weights[#i];
                #valid &= #id_i < #new_s_i;
            }
        } else {
            quote_spanned! { span =>
                #gen_global_id
                let #new_s_i = #new_s[#i] as u32;
                let #old_s_i = #old_s[#i] as u32;
                #global_idx += #id_i as usize * #weights[#i];
                #valid &= #id_i < #new_s_i;
            }
        };
        // avoid warning: value assigned to `remain_id` is never read
        if i != local_id_len - 1 {
            gen_global_id = quote_spanned! { span =>
                #gen_global_id
                let #remain_id = #remain_id / #old_s_i;
            };
        }
    }
    // ensure all tid are consumed
    gen_global_id = quote_spanned! { span =>
        #gen_global_id
        #valid &= #remain_tid == 0;
    };
    let gpu_crate = args.gpu_crate;
    quote_spanned! { span => {
        #[derive(Clone, Copy)]
        struct PrivateMapReshapeShuffle{
            old_sizes: [usize; #len],
            new_sizes: [usize; #len],
            offset: usize,
            cached_weights: [usize; #len],
        }
        unsafe impl<CS: #gpu_crate::chunk_scope::ChunkScope> #gpu_crate::chunk::ScopeUniqueMap<CS> for PrivateMapReshapeShuffle
        {
            type IndexType = usize;

            fn map(&self, #lid: usize, thread_ids: [u32; #gpu_crate::chunk_scope::TID_MAX_LEN]) -> (bool, usize) {
                 let #tid = CS::global_id_x(thread_ids)
            + CS::global_dim_x()
                * (CS::global_id_y(thread_ids) + CS::global_dim_y() * CS::global_id_z(thread_ids));
            let #old_s = self.old_sizes;
            let #new_s = self.new_sizes;
            let #weights = self.cached_weights;
            #gen_global_id
            (#valid, #global_idx)
        }
    }
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
        Expr::Lit(expr_lit) | Expr::Group(ExprGroup { expr: box Expr::Lit(expr_lit), .. }) => {
            if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                let val = lit_int.base10_parse::<usize>().unwrap();
                Some(val)
            } else {
                None
            }
        }
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
            Expr::Unary(ExprUnary { op: syn::UnOp::Neg(_), box expr, .. }) => {
                (get_const_val(expr), true)
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
