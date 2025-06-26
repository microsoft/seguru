use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::TokenTree;
use rustc_hir::Attribute;
use rustc_hir::def_id::DefId;
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
    BlockDim,
    GridDim,
    Printf,
    AddStringAttr,
    Scope,
    Grid,
    Block,
    Launch,
    IntoIter,
    IterNext,
    Subslice,
    SubsliceMut,
}

impl TryFrom<&str> for GpuItem {
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let ret = match s {
            "gpu.thread_id" => GpuItem::ThreadId,
            "gpu.global_thread_id" => GpuItem::GlobalThreadId,
            "gpu.block_dim" => GpuItem::BlockDim,
            "gpu.grid_dim" => GpuItem::GridDim,
            "gpu.printf" => GpuItem::Printf,
            "gpu.scope" => GpuItem::Scope,
            "gpu.block" => GpuItem::Block,
            "gpu.grid" => GpuItem::Grid,
            "gpu.launch" => GpuItem::Launch,
            "add_mlir_string_attr" => GpuItem::AddStringAttr,
            "gpu.into_iter" => GpuItem::IntoIter,
            "gpu.next" => GpuItem::IterNext,
            "gpu.subslice" => GpuItem::Subslice,
            "gpu.subslice_mut" => GpuItem::SubsliceMut,
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
            GpuItem::BlockDim => "gpu.block_dim",
            GpuItem::GridDim => "gpu.grid_dim",
            GpuItem::Printf => "gpu.printf",
            GpuItem::AddStringAttr => "add_mlir_string_attr",
            GpuItem::Scope => "gpu.scope",
            GpuItem::Grid => "gpu.grid",
            GpuItem::Block => "gpu.block",
            GpuItem::Launch => "gpu.launch",
            GpuItem::IntoIter => "gpu.into_iter",
            GpuItem::IterNext => "gpu.next",
            GpuItem::Subslice => "gpu.subslice",
            GpuItem::SubsliceMut => "gpu.subslice_mut",
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
    pub fn build(tcx: &rustc_middle::ty::TyCtxt<'_>, def_id: DefId) -> GpuAttributes {
        let attrs = tcx.get_attrs_unchecked(def_id);
        GpuAttributes::parse(attrs)
    }

    pub fn to_mlir_attribute<'ml>(
        &self,
        ctx: &'ml melior::Context,
    ) -> Option<melior::ir::Attribute<'ml>> {
        self.gpu_item.map(|gpu_item| {
            melior::ir::attribute::StringAttribute::new(ctx, gpu_item.into()).into()
        })
    }

    pub fn is_gpu_related(&self) -> bool {
        self.kernel || self.host || self.device || self.gpu_item.is_some()
    }

    pub fn is_builtin(&self) -> bool {
        self.gpu_item.is_some()
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
                gpu_attrs.gpu_item = Some(
                    sym.as_str()
                        .try_into()
                        .unwrap_or_else(|_| panic!("Unsupported builtin item {}", sym.as_str())),
                );
            }
        }
        gpu_attrs
    }
}
