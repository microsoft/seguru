use proc_macro::TokenStream;
use quote::quote_spanned;
use syn::parse::Parse;
use syn::{Expr, ExprArray, ExprGroup, Ident, Path};

// map!([4,4] | [2,2] => weights: [1, 4, 10, 1000])
// map!([4,4] | [2,2] => layout: [0, 1, 2, 3])
struct MapArgs {
    span: proc_macro2::Span,
    gpu_crate: Path,
    _comma: syn::Token![,],
    sizes: ExprArray,
    _semi: syn::Token![|],
    extra_sizes: ExprArray,
    _holder: syn::Token![=>],
    _kind: Ident,
    weights: Option<ExprArray>,
    layout: Option<ExprArray>,
}

impl Parse for MapArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let gpu_crate = input.parse()?;
        let _comma = input.parse()?;
        let sizes = input.parse()?;
        let _semi: syn::Token![|] = input.parse()?;
        let extra_sizes = input.parse()?;
        let _holder: syn::Token![=>] = input.parse()?;
        let _kind = input.parse()?;
        let weights = if _kind == "weights" {
            let _: syn::Token![:] = input.parse()?;
            let weights = input.parse()?;
            Some(weights)
        } else {
            None
        };
        let layout = if _kind == "layout" {
            let _: syn::Token![:] = input.parse()?;
            let layout: ExprArray = input.parse()?;
            Some(layout)
        } else {
            None
        };
        assert!(weights.is_some() || layout.is_some(), "Expected either weights or layout");
        Ok(MapArgs {
            span: input.span(),
            gpu_crate,
            _comma,
            sizes,
            _semi,
            extra_sizes,
            _holder,
            _kind,
            weights,
            layout,
        })
    }
}

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
                format!("duplicate value {}", v),
            ));
        }
        seen[v] = true;
    }
    Ok(())
}

fn get_const_array(expr: impl Iterator<Item = Expr>) -> Result<Vec<usize>, syn::Error> {
    let mut const_array = vec![];
    for size in expr {
        if let Expr::Lit(expr_lit) | Expr::Group(ExprGroup { expr: box Expr::Lit(expr_lit), .. }) =
            size
        {
            if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                let val = lit_int.base10_parse::<usize>().unwrap();
                const_array.push(val);
            } else {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("Expected integer literal in sizes array {:?}", expr_lit),
                ));
            }
        } else {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("Expected integer literal in sizes array {:?}", size),
            ));
        }
    }
    Ok(const_array)
}

#[allow(clippy::type_complexity)]
fn check_weight(
    sizes: &ExprArray,
    extra_sizes: &ExprArray,
    weights: &ExprArray,
) -> Result<Option<(Vec<usize>, Vec<usize>, usize, Vec<usize>, Vec<usize>, usize)>, syn::Error> {
    let const_weight = match get_const_array(weights.elems.iter().cloned()) {
        Ok(w) => w,
        Err(_) => return Ok(None), // Not constant, skip check
    };
    let nsize = sizes.elems.len();
    let msize = extra_sizes.elems.len();
    let const_sizes =
        match get_const_array(sizes.elems.iter().chain(extra_sizes.elems.iter()).cloned()) {
            Ok(s) => s,
            Err(_) => return Ok(None), // Not constant, skip check
        };
    let mut old_tid_weights = vec![];
    for i in 0..sizes.elems.len() {
        old_tid_weights.push(const_sizes[i + 1..].iter().product::<usize>());
    }
    let max_tid = const_sizes[..nsize].iter().product::<usize>();
    let mut old_idx_weights = vec![];
    for i in nsize..const_sizes.len() {
        old_idx_weights.push(const_sizes[i + 1..].iter().product::<usize>());
    }
    let max_idx = const_sizes[nsize..].iter().product::<usize>();
    if const_weight.len() != const_sizes.len() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "weight array length ({}) must match sizes length {}",
                const_weight.len(),
                nsize + msize
            ),
        ));
    }
    // Check all weights are non-zero
    for (i, &w) in const_weight.iter().enumerate() {
        let mut sum = 0;
        for j in 0..const_weight.len() {
            if i != j && const_weight[j] <= w {
                sum += const_weight[j] * (const_sizes[j] - 1);
            }
        }
        if sum >= w && const_sizes[0] > 1 {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "weight {} too small, other can sum up to  {}: {:?} {:?}",
                    w, sum, const_sizes, const_weight
                ),
            ));
        }
    }
    Ok(Some((
        old_tid_weights,
        const_weight[..nsize].to_vec(),
        max_tid,
        old_idx_weights,
        const_weight[nsize..].to_vec(),
        max_idx,
    )))
}

pub fn reshape_map(input: TokenStream) -> TokenStream {
    let args: MapArgs = syn::parse(input)
        .expect("Map args must be of the form map!(gpu_crate, [sizes] | [extra_sizes] => weights: [...] or layout: [...])");

    let sizes = args.sizes;
    let extra_sizes = args.extra_sizes;
    let nsize = sizes.elems.len();
    if nsize < 1 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "thread index sizes must have at least one dimension",
        )
        .to_compile_error()
        .into();
    }
    let msize = extra_sizes.elems.len();
    if msize < 1 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "extra index sizes must have at least one dimension",
        )
        .to_compile_error()
        .into();
    }

    let span = args.span;
    let gpu_crate = args.gpu_crate;
    let def = quote_spanned! {
        span =>
        #[derive(Clone, Copy)]
        struct PrivateReshapedMap(#gpu_crate::MapReshape<#nsize, #msize>);
        // #Safety
        // This is safe because the inner map is safe.
        unsafe impl<CS: #gpu_crate::chunk_scope::ChunkScope>
            #gpu_crate::chunk::ScopeUniqueMap<CS> for PrivateReshapedMap
        {
            type IndexType = usize;

            #[inline]
            #[gpu_macros::device]
            fn map(
                &self,
                idx: Self::IndexType,
                thread_ids: [usize; #gpu_crate::chunk_scope::TID_MAX_LEN],
            ) -> (bool, usize) {
                self.0.map::<CS>(idx, thread_ids)
            }
        }
    };
    let mut all_sizes = sizes.clone();
    all_sizes.elems.extend(extra_sizes.elems.iter().cloned());
    if let Some(weights) = args.weights {
        if weights.elems.len() != msize + nsize {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("weight array length ({}) must match thread index sizes {} + extra index sizes {}", weights.elems.len(), nsize, msize),
            )
            .to_compile_error()
            .into();
        }
        // Check weights validity if it is constant.
        let check = check_weight(&sizes, &extra_sizes, &weights);

        let inner = match check {
            Ok(Some((
                old_tid_weights,
                const_weight,
                max_tid,
                old_idx_weights,
                const_weight_idx,
                max_idx,
            ))) => {
                quote_spanned! {
                    span => {
                        // This is safe because we have checked the weights at compile time.
                        let inner = gpu::MapReshape::<#nsize, #msize>::new_with_weight_no_check(
                            [#(#old_tid_weights),*],
                            [#(#const_weight),*],
                            [#(#old_idx_weights),*],
                            [#(#const_weight_idx),*],
                            #max_tid,
                            #max_idx,
                        );

                        PrivateReshapedMap(inner)
                    }
                }
            }
            Ok(None) => {
                quote_spanned! {
                    span => {
                        // This is dynamically checked to be safe at runtime.
                        let inner = #gpu_crate::MapReshape::<#nsize, #msize>::new_with_weight(#all_sizes, #weights);
                        PrivateReshapedMap(inner)
                    }
                }
            }
            Err(e) => return e.to_compile_error().into(),
        };

        quote_spanned! {
            span => {
                #def
                #inner
            }
        }
        .into()
    } else if let Some(layout) = args.layout {
        let const_layout = match get_const_array(layout.elems.iter().cloned()) {
            Ok(l) => l,
            Err(e) => return e.to_compile_error().into(),
        };
        if let Err(e) = check_permutation(&const_layout) {
            e.to_compile_error().into()
        } else {
            quote_spanned! {
                span => {
                    #def
                    let inner = #gpu_crate::MapReshape::<#nsize, #msize>::new(#all_sizes, #layout);
                    PrivateReshapedMap(inner)
                }
            }
            .into()
        }
    } else {
        syn::Error::new(proc_macro2::Span::call_site(), "Must provide either weights or layout")
            .to_compile_error()
            .into()
    }
}
