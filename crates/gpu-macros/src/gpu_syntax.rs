use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote_spanned};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{Expr, Path, parse_quote};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostState {
    Scope,
    Grid,
    Block,
    None,
}
pub struct GpuFunctionRewriter {
    inside_host: Option<HostState>,
    inside_device_entry: bool,
    inside_device: bool,
    grid_loop_span: Option<proc_macro2::Span>,
    block_loop_span: Option<proc_macro2::Span>,
}

fn path_matches(path: &Path, target: &str) -> bool {
    let segs = target.split("::").collect::<Vec<_>>();
    let segments: Vec<_> = path.segments.iter().map(|seg| seg.ident.to_string()).collect();
    segments == segs
}

fn expr_path_matches(expr: &Expr, target: &str) -> Option<bool> {
    if let Expr::Path(path) = expr { Some(path_matches(&path.path, target)) } else { None }
}

fn span_to_tokens(span: Span, to_span: Span) -> proc_macro2::TokenStream {
    let start = span.start();
    let end = span.end();
    let to_lit = |val| proc_macro2::Literal::u32_unsuffixed(val as u32);
    let line_start = to_lit(start.line);
    let col_start = to_lit(start.column + 1); // why +1?
    let line_end = to_lit(end.line);
    let col_end = to_lit(end.column + 1);
    quote_spanned! {to_span =>
        #line_start, #col_start, #line_end, #col_end
    }
}

impl GpuFunctionRewriter {
    fn get_gpu_loop_expr_mut(&self, expr: &mut syn::Expr, path_str: &str) -> Option<Span> {
        if let Expr::ForLoop(fl) = expr {
            if let Expr::Call(syn::ExprCall { func, .. }) = fl.expr.as_ref() {
                if let Some(true) = expr_path_matches(&func, path_str) {
                    return Some(fl.span());
                }
            }
        }
        None
    }
}

impl VisitMut for GpuFunctionRewriter {
    fn visit_expr_closure_mut(&mut self, closure: &mut syn::ExprClosure) {
        syn::visit_mut::visit_expr_closure_mut(self, closure);
        if !closure.attrs.iter().any(|a| path_matches(a.path(), "gpu_codegen::device")) {
            closure.attrs.push(parse_quote!(#[gpu_codegen::device]));
        }
        if let Some(span) = self.grid_loop_span {
            let span_tokens = span_to_tokens(span, closure.span());
            closure.attrs.push(parse_quote!(#[gpu_codegen::grid_loop(#span_tokens)]));
        }
        if let Some(span) = self.block_loop_span {
            let span_tokens = span_to_tokens(span, closure.span());
            closure.attrs.push(parse_quote!(#[gpu_codegen::block_loop(#span_tokens)]));
        }
    }

    fn visit_expr_call_mut(&mut self, call: &mut syn::ExprCall) {
        let prev_inside_host = self.inside_host;
        match self.inside_host {
            Some(HostState::None)
                if matches!(expr_path_matches(&call.func, "gpu::scope"), Some(true)) =>
            {
                self.inside_host = Some(HostState::Scope);
            }
            Some(_) if matches!(expr_path_matches(&call.func, "gpu::scope"), Some(true)) => {
                panic!("scope calls are not allowed in this context");
            }
            _ => {}
        }
        syn::visit_mut::visit_expr_call_mut(self, call);
        self.inside_host = prev_inside_host;
    }

    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        let prev_inside_host = self.inside_host;
        match self.inside_host {
            Some(HostState::Scope) if matches!(expr, Expr::ForLoop(_)) => {
                if let Some(span) = self.get_gpu_loop_expr_mut(expr, "gpu::grid") {
                    self.grid_loop_span = Some(span);
                    self.inside_host = Some(HostState::Grid);
                } else {
                    panic!("scope must use a grid loop");
                }
            }
            Some(HostState::Grid) if matches!(expr, Expr::ForLoop(_)) => {
                if let Some(span) = self.get_gpu_loop_expr_mut(expr, "gpu::block") {
                    self.block_loop_span = Some(span);
                    self.inside_host = Some(HostState::Block);
                } else {
                    panic!("The grid loop must be have a block loops");
                }
            }
            _ => {}
        }
        syn::visit_mut::visit_expr_mut(self, expr);
        self.inside_host = prev_inside_host;
    }

    fn visit_item_fn_mut(&mut self, f: &mut syn::ItemFn) {
        syn::visit_mut::visit_item_fn_mut(self, f);
        if self.inside_host.is_some() {
            // If we are inside a host function, we need to ensure it has the #[gpu_codegen::host] attribute
            if !f.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::host")) {
                f.attrs.push(parse_quote!(#[gpu_codegen::host]));
            }
        } else if self.inside_device_entry {
            // If we are inside a device entry function, we need to ensure it has the #[gpu_codegen::device_entry] attribute
            if !f.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::device_entry")) {
                f.attrs.push(parse_quote!(#[gpu_codegen::device_entry]));
            }
        } else if self.inside_device {
            // If we are inside a device function, we need to ensure it has the #[gpu_codegen::device] attribute
            if !f.attrs.iter().any(|a| a.path().is_ident("gpu_codegen::device")) {
                f.attrs.push(parse_quote!(#[gpu_codegen::device]));
            }
        }
    }
}

pub(crate) fn rewrite_host(input: TokenStream) -> TokenStream {
    let mut visitor = GpuFunctionRewriter {
        inside_host: Some(HostState::None),
        inside_device_entry: false,
        inside_device: false,
        block_loop_span: None,
        grid_loop_span: None,
    };
    let mut f = syn::parse_macro_input!(input as syn::ItemFn);
    visitor.visit_item_fn_mut(&mut f);
    f.to_token_stream().into()
}
