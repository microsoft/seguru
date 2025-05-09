use rustc_hir::{def_id::DefId, Attribute};
use rustc_span::Symbol;

// inspired by rust-gpu's attribute handling
#[derive(Default, Clone, PartialEq)]
pub(crate) struct GpuAttributes {
    pub kernel: bool,
}

pub fn gpu_symbol() -> Symbol {
    Symbol::intern("gpu_codegen")
}

pub fn kernel_symbol() -> Symbol {
    Symbol::intern("kernel")
}

impl GpuAttributes {
    pub fn parse<'tcx>(attrs: &'tcx [Attribute]) -> Self {
        let mut gpu_attrs = Self::default();

        for attr in attrs {
            if attr.path_matches(&[gpu_symbol(), kernel_symbol()]) {
                gpu_attrs.kernel = true;
            }
        }
        gpu_attrs
    }
}

pub(crate) fn is_gpu_code(tcx: &rustc_middle::ty::TyCtxt<'_>, def_id: DefId) -> bool {
    let attrs = tcx.get_attrs_unchecked(def_id);
    let gpu_attrs = GpuAttributes::parse(attrs);
    if gpu_attrs.kernel {
        true
    } else {
        false
    }
}
