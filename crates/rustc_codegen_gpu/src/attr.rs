use rustc_ast::{
    token::{Token, TokenKind},
    tokenstream::TokenTree,
};
use rustc_hir::{def_id::DefId, Attribute};
use rustc_span::Symbol;

// inspired by rust-gpu's attribute handling
#[derive(Default, Clone, PartialEq)]
pub(crate) struct GpuAttributes {
    pub kernel: bool,
    pub host: bool,
    pub device: bool, // a device function called by a kernel but not by host directly
    pub gpu_item: Option<GpuItem>,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum GpuItem {
    ThreadId,
    GlobalThreadId,
    Printf,
    AddStringAttr,
}

impl TryFrom<&str> for GpuItem {
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let ret = match s {
            "gpu.thread_id" => GpuItem::ThreadId,
            "gpu.global_thread_id" => GpuItem::GlobalThreadId,
            "gpu.printf" => GpuItem::Printf,
            "add_mlir_string_attr" => GpuItem::AddStringAttr,
            _ => return Err(()),
        };
        Ok(ret)
    }

    type Error = ();
}

impl From<GpuItem> for &'static str {
    fn from(item: GpuItem) -> &'static str {
        match item {
            GpuItem::ThreadId => "gpu.thread_id",
            GpuItem::GlobalThreadId => "gpu.global_thread_id",
            GpuItem::Printf => "gpu.printf",
            GpuItem::AddStringAttr => "add_mlir_string_attr",
        }
    }
}

pub fn gpu_symbol() -> Symbol {
    Symbol::intern("gpu_codegen")
}

pub fn kernel_symbol() -> Symbol {
    Symbol::intern("kernel")
}

pub fn host_symbol() -> Symbol {
    Symbol::intern("host")
}

pub fn device_symbol() -> Symbol {
    Symbol::intern("device")
}

pub fn gpu_builtin_symbol() -> Symbol {
    Symbol::intern("builtin")
}

pub(crate) fn token_to_string(token: &Token) -> Result<Option<String>, ()> {
    match token.kind {
        TokenKind::Literal(lit) => Ok(Some(lit.symbol.as_str().to_string())),
        TokenKind::Ident(symbol, _) => Ok(Some(symbol.as_str().to_string())),
        TokenKind::Comma => Ok(None),
        TokenKind::Dot => Ok(Some(".".to_string())),
        _ => Err(()),
    }
}

impl GpuAttributes {
    pub fn to_mlir_attribute<'ml>(
        &self,
        ctx: &'ml melior::Context,
    ) -> Option<melior::ir::Attribute<'ml>> {
        self.gpu_item.map(|gpu_item| {
            melior::ir::attribute::StringAttribute::new(ctx, gpu_item.into()).into()
        })
    }
    pub fn parse(attrs: &[Attribute]) -> Self {
        let mut gpu_attrs = Self::default();

        for attr in attrs {
            if attr.path_matches(&[gpu_symbol(), kernel_symbol()]) {
                gpu_attrs.kernel = true;
            }
            if attr.path_matches(&[gpu_symbol(), host_symbol()]) {
                gpu_attrs.host = true;
            }
            if attr.path_matches(&[gpu_symbol(), device_symbol()]) {
                gpu_attrs.device = true;
            }
            if attr.path_matches(&[gpu_symbol(), gpu_builtin_symbol()]) {
                let Attribute::Unparsed(item) = attr else {
                    dbg!(attr);
                    panic!("gpu builtin func must have a name");
                };
                let sym = match &item.args {
                    rustc_hir::AttrArgs::Delimited(delim_args) => delim_args
                        .tokens
                        .iter()
                        .map(|tt| match tt {
                            TokenTree::Token(t, _) => token_to_string(t).unwrap().unwrap(),
                            TokenTree::Delimited(
                                delim_span,
                                delim_spacing,
                                delimiter,
                                token_stream,
                            ) => todo!(),
                        })
                        .collect::<Vec<_>>()
                        .concat(),
                    _ => panic!("gpu builtin func must have a name"),
                };
                gpu_attrs.gpu_item = Some(sym.as_str().try_into().unwrap());
            }
        }
        gpu_attrs
    }
}

pub(crate) fn is_gpu_code(tcx: &rustc_middle::ty::TyCtxt<'_>, def_id: DefId) -> bool {
    let attrs = tcx.get_attrs_unchecked(def_id);
    let gpu_attrs = GpuAttributes::parse(attrs);
    gpu_attrs.kernel || gpu_attrs.host || gpu_attrs.device
}
