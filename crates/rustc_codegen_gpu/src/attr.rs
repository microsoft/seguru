use rustc_hir::def_id::DefId;
use rustc_hir::{Attribute, LangItem};
use rustc_span::Symbol;

use crate::mlir::gpu::NvmmLaunchBound;

// inspired by rust-gpu's attribute handling
#[derive(Default, Clone, PartialEq)]
pub(crate) struct GpuAttributes {
    pub kernel: bool,
    pub host: bool,
    pub device: bool, // a device function called by a kernel but not by host directly
    pub gpu_item: Option<GpuItem>,

    // [0..MAX_FN_IN_PARAMS] indicates the parameters that are in shared memory space.
    // [MAX_FN_IN_PARAMS..MAX] indicates the return value is in shared memory space.
    pub shared_data: Vec<usize>,

    pub shared_size: bool, // this is used to decide whether to call `gpu_shared_size`.

    // The parameters that should have same value across threads.
    // If empty, it requires the control flow is the same across threads.
    // If not empty, it requires both control flow and data flow (for arguments
    // at indices) are the same across threads.
    pub sync_data: Option<Vec<usize>>,

    // Indices of parameters that must have the same value across all threads.
    // [0..MAX_FN_IN_PARAMS] indices indicate the mutable parameters.
    // Use [MAX_FN_IN_PARAMS..max] to indicate the return value.
    // Ensures that if these inputs are uniform, the return value
    // will also be uniform across GPU threads.
    // No effect if empty.
    pub ret_sync_data: Vec<usize>,
    // disable diverse checker for this function
    // Due to the limitation of our current diverse checker,
    // some functions need to disable the diverse checker.
    pub disable_diverse_checker: bool,

    pub launch_bound: Option<NvmmLaunchBound>,
}

impl GpuAttributes {
    pub const MAX_FN_IN_PARAMS: usize = 1000;
}

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum GpuItem {
    AllReduce,
    ThreadId,
    GlobalThreadId,
    LaneId,
    NvvmReduxSync,
    SubgroupId,
    SubgroupReduce,
    SubgroupSize,
    Shuffle,
    BlockDim,
    BlockId,
    GridDim,
    PrintFormat,
    PrintArgs,
    AddStringAttr,
    Scope,
    Launch,
    NewChunk,
    UniqueChunk,
    SyncThreads, // this is used to decide whether to call thread_sync.
    Subslice,
    SubsliceMut,
    NewSharedMem,
    AtomicRMW,
    BuildSFI,
    GetLocalMut2D,
    GetLocal2D,
    DynamicShared,
    DeviceIntrinsic(String),
    Core(LangItem), // for core crate items
    CoreFn(String), // for core crate functions
    DiagnoseOnly(String),
}

fn lang_item_from_str(name: &str) -> Option<LangItem> {
    tracing::debug!("lang_item_from_str: {}", name);
    LangItem::from_name(Symbol::intern(name))
}

const PANIC_FUNCTIONS: [&str; 8] = [
    "core::slice::index::slice_index_order_fail",
    "core::slice::index::slice_start_index_len_fail",
    "core::slice::index::slice_end_index_len_fail",
    "core::option::unwrap_failed",
    "std::slice::index::slice_index_order_fail",
    "std::slice::index::slice_start_index_len_fail",
    "std::slice::index::slice_end_index_len_fail",
    "std::option::unwrap_failed",
];

const PANIC_FUNCTION_PATTERNS: [&str; 2] = [
    r"core::slice::.*::copy_from_slice::len_mismatch_fail",
    r"std::slice::.*::copy_from_slice::len_mismatch_fail",
];

pub fn is_panic_function(path: &str) -> bool {
    let is_panic = PANIC_FUNCTIONS.contains(&path);
    if is_panic {
        return true;
    }
    for pattern in PANIC_FUNCTION_PATTERNS {
        let re = regex::Regex::new(pattern).unwrap();
        if re.is_match(path) {
            return true;
        }
    }
    false
}

impl TryFrom<&str> for GpuItem {
    type Error = ();

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let ret = match s {
            "gpu::all_reduce" => GpuItem::AllReduce,
            "gpu::thread_id" => GpuItem::ThreadId,
            "gpu::global_thread_id" => GpuItem::GlobalThreadId,
            "gpu::lane_id" => GpuItem::LaneId,
            "gpu::subgroup_id" => GpuItem::SubgroupId,
            "gpu::subgroup_reduce" => GpuItem::SubgroupReduce,
            "gpu::subgroup_size" => GpuItem::SubgroupSize,
            "gpu::shuffle" => GpuItem::Shuffle,
            "nvvm::redux_sync" => GpuItem::NvvmReduxSync,
            "gpu::block_dim" => GpuItem::BlockDim,
            "gpu::block_id" => GpuItem::BlockId,
            "gpu::grid_dim" => GpuItem::GridDim,
            "gpu::printf" => GpuItem::PrintFormat,
            "gpu::print_args" => GpuItem::PrintArgs,
            "gpu::scope" => GpuItem::Scope,
            "gpu::launch" => GpuItem::Launch,
            "gpu::add_mlir_string_attr" => GpuItem::AddStringAttr,
            "gpu::GpuChunksMut::new" => GpuItem::NewChunk,
            "gpu::GpuChunksMut::unique_chunk" => GpuItem::UniqueChunk,
            "gpu::sync_threads" => GpuItem::SyncThreads,
            "gpu::subslice" => GpuItem::Subslice,
            "gpu::subslice_mut" => GpuItem::SubsliceMut,
            "gpu::new_shared_mem" => GpuItem::NewSharedMem,
            "gpu::atomic_rmw" => GpuItem::AtomicRMW,
            "gpu::build_sfi" => GpuItem::BuildSFI,
            "gpu::get_local_mut_2d" => GpuItem::GetLocalMut2D,
            "gpu::get_local_2d" => GpuItem::GetLocal2D,
            "gpu::base_dynamic_shared" => GpuItem::DynamicShared,
            s if s.starts_with("gpu::device_intrinsics::") => {
                GpuItem::DeviceIntrinsic(s.replace("gpu::device_intrinsics::", ""))
            }
            // Override cmath functions to use our device intrinsics
            "std::sys::cmath::{extern#0}::tanhf" => {
                GpuItem::DeviceIntrinsic("gpu::device_intrinsics::tanh".into())
            }
            "std::sys::cmath::{extern#0}::tanf" => {
                GpuItem::DeviceIntrinsic("gpu::device_intrinsics::tanf".into())
            }
            "std::sys::cmath::{extern#0}::coshf" => {
                GpuItem::DeviceIntrinsic("gpu::device_intrinsics::cosh".into())
            }
            "std::sys::cmath::{extern#0}::sinhf" => {
                GpuItem::DeviceIntrinsic("gpu::device_intrinsics::sinh".into())
            }
            s if is_panic_function(s) => GpuItem::CoreFn(s.to_string()),
            s if s.starts_with("core") || s.starts_with("std") => {
                if let Some(i) = lang_item_from_str(s) {
                    GpuItem::Core(i)
                } else {
                    return Err(());
                }
            }
            _ => {
                if let Some(i) = lang_item_from_str(s) {
                    GpuItem::Core(i)
                } else {
                    return Err(());
                }
            }
        };
        Ok(ret)
    }
}

impl From<GpuItem> for String {
    fn from(item: GpuItem) -> String {
        match item {
            GpuItem::AllReduce => "gpu::all_reduce".into(),
            GpuItem::ThreadId => "gpu::thread_id".into(),
            GpuItem::GlobalThreadId => "gpu::global_thread_id".into(),
            GpuItem::LaneId => "gpu::lane_id".into(),
            GpuItem::SubgroupId => "gpu::subgroup_id".into(),
            GpuItem::SubgroupReduce => "gpu::subgroup_reduce".into(),
            GpuItem::SubgroupSize => "gpu::subgroup_size".into(),
            GpuItem::Shuffle => "gpu::shuffle".into(),
            GpuItem::NvvmReduxSync => "nvvm::redux_sync".into(),
            GpuItem::BlockDim => "gpu::block_dim".into(),
            GpuItem::BlockId => "gpu::block_id".into(),
            GpuItem::GridDim => "gpu::grid_dim".into(),
            GpuItem::PrintFormat => "gpu::printf".into(),
            GpuItem::PrintArgs => "gpu::print_args".into(),
            GpuItem::AddStringAttr => "gpu::add_mlir_string_attr".into(),
            GpuItem::Scope => "gpu::scope".into(),
            GpuItem::Launch => "gpu::launch".into(),
            GpuItem::NewChunk => "gpu::GpuChunksMut::new".into(),
            GpuItem::UniqueChunk => "gpu::GpuChunksMut::unique_chunk".into(),
            GpuItem::SyncThreads => "gpu::sync_threads".into(),
            GpuItem::Subslice => "gpu::subslice".into(),
            GpuItem::SubsliceMut => "gpu::subslice_mut".into(),
            GpuItem::NewSharedMem => "gpu::new_shared_mem".into(),
            GpuItem::AtomicRMW => "gpu::atomic_rmw".into(),
            GpuItem::BuildSFI => "gpu::build_sfi".into(),
            GpuItem::GetLocalMut2D => "gpu::get_local_mut_2d".into(),
            GpuItem::GetLocal2D => "gpu::get_local_2d".into(),
            GpuItem::DynamicShared => "gpu::base_dynamic_shared".into(),
            GpuItem::DeviceIntrinsic(name) => {
                format!("gpu::device_intrinsics::{}", name)
            }
            GpuItem::Core(name) => name.name().to_string(),
            GpuItem::CoreFn(name) => name,
            GpuItem::DiagnoseOnly(name) => name,
        }
    }
}

impl From<GpuItem> for Symbol {
    fn from(item: GpuItem) -> Self {
        Symbol::intern(&String::from(item))
    }
}

impl TryFrom<Symbol> for GpuItem {
    type Error = ();

    fn try_from(symbol: Symbol) -> Result<Self, Self::Error> {
        let s = symbol.as_str();
        GpuItem::try_from(s)
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

pub fn gpu_shared_size_symbol() -> Symbol {
    Symbol::intern("shared_size")
}

pub fn memspace_shared() -> Symbol {
    // 0..MAX_FN_IN_PARAMS for parameters, MAX_FN_IN_PARAMS..max for return value
    Symbol::intern("memspace_shared")
}

pub fn sync_data() -> Symbol {
    Symbol::intern("sync_data")
}

fn ret_sync_data() -> Symbol {
    Symbol::intern("ret_sync_data")
}

pub fn nvvm_launch_bound() -> Symbol {
    Symbol::intern("nvvm_launch_bound")
}

impl GpuAttributes {
    pub fn callee_device() -> Self {
        GpuAttributes { device: true, ..GpuAttributes::default() }
    }

    pub fn build(tcx: &rustc_middle::ty::TyCtxt<'_>, def_id: DefId) -> GpuAttributes {
        let attrs = tcx.get_attrs_unchecked(def_id);
        let mut gpu_attr = GpuAttributes::parse(attrs);
        let crate_name = tcx.crate_name(def_id.krate);
        if crate_name == rustc_span::sym::core || crate_name == rustc_span::sym::std {
            if let Some(lang_item) = tcx.lang_items().from_def_id(def_id) {
                if lang_item.name().to_string().starts_with("panic") {
                    gpu_attr.device = true;
                    gpu_attr.gpu_item = Some(GpuItem::Core(lang_item));
                }
            } else {
                // def_path_str might be re-exported, so we use to_string_no_crate_verbose.
                let path =
                    format!("{}{}", crate_name, tcx.def_path(def_id).to_string_no_crate_verbose());
                //eprintln!("Checking core/std item: {}", path);
                let path: &str = &path;
                if let Ok(gpu_item) = GpuItem::try_from(path) {
                    gpu_attr.device = true;
                    gpu_attr.gpu_item = Some(gpu_item);
                }
            }
        }
        if let Some(sym) = tcx.all_diagnostic_items(()).id_to_name.get(&def_id) {
            if let Ok(gpu_item) = GpuItem::try_from(*sym) {
                gpu_attr.gpu_item = Some(gpu_item);
            } else {
                gpu_attr.gpu_item = Some(GpuItem::DiagnoseOnly(sym.as_str().into()));
            }
        }

        gpu_attr
    }

    pub fn to_mlir_attributes<'ml>(
        &self,
        ctx: &'ml melior::Context,
    ) -> Vec<(&'static str, melior::ir::Attribute<'ml>)> {
        let mut ret = vec![];
        if self.is_builtin() {
            ret.push((
                crate::mlir::BUILTIN_SYM,
                self.gpu_item
                    .clone()
                    .map(|gpu_item| {
                        melior::ir::attribute::StringAttribute::new(ctx, &String::from(gpu_item))
                            .into()
                    })
                    .unwrap(),
            ));
        }
        if let Some(bound) = &self.launch_bound {
            ret.extend(bound.to_attrs(ctx));
        }
        ret
    }

    pub fn is_gpu_related(&self) -> bool {
        self.kernel
            || self.host
            || self.device
            || !matches!(self.gpu_item, Some(GpuItem::DiagnoseOnly(_)) | None)
            || self.shared_size
    }

    pub fn is_builtin(&self) -> bool {
        match &self.gpu_item {
            Some(GpuItem::NewChunk) => false,
            Some(GpuItem::GetLocalMut2D) => false,
            Some(GpuItem::GetLocal2D) => false,
            Some(GpuItem::Core(lang_item)) if lang_item.name().to_string().starts_with("panic") => {
                true
            }
            Some(GpuItem::CoreFn(path)) if is_panic_function(path.as_str()) => true,
            Some(GpuItem::DiagnoseOnly(_)) => false,
            Some(_) => true,
            _ => false,
        }
    }

    fn parse(attrs: &[Attribute]) -> Self {
        let mut gpu_attrs = Self::default();

        let get_meta_usize_list = |attr: &Attribute| {
            let mut ret_sync_data = vec![];
            if let Some(meta_list) = attr.meta_item_list() {
                // Expect a literal like 1, 2, -1
                for meta in meta_list {
                    if let Some(lit) = meta.lit() {
                        if let rustc_ast::LitKind::Int(value, int_ty) = lit.kind {
                            ret_sync_data.push(value.get() as usize);
                        } else {
                            panic!("Expected an integer, found {:?} at {:?}", lit, attr.span());
                        }
                    } else {
                        panic!("Expected a literal, found {:?} at {:?}", meta, attr.span());
                    }
                }
            }
            ret_sync_data
        };

        for attr in attrs {
            if attr.path_matches(&[gpu_symbol(), kernel_symbol()]) {
                gpu_attrs.kernel = true;
            } else if attr.path_matches(&[gpu_symbol(), host_symbol()]) {
                gpu_attrs.host = true;
            } else if attr.path_matches(&[gpu_symbol(), device_symbol()]) {
                gpu_attrs.device = true;
            } else if attr.path_matches(&[gpu_symbol(), gpu_shared_size_symbol()]) {
                gpu_attrs.shared_size = true;
            } else if attr.path_matches(&[gpu_symbol(), memspace_shared()]) {
                let shared_data = get_meta_usize_list(attr);
                gpu_attrs.shared_data = shared_data;
            } else if attr.path_matches(&[gpu_symbol(), sync_data()]) {
                let sync_data = get_meta_usize_list(attr);
                gpu_attrs.sync_data = Some(sync_data);
            } else if attr.path_matches(&[gpu_symbol(), ret_sync_data()]) {
                let ret_sync_data = get_meta_usize_list(attr);
                gpu_attrs.ret_sync_data = ret_sync_data;
            } else if attr.path_matches(&[gpu_symbol(), Symbol::intern("skip_divergence_check")]) {
                gpu_attrs.disable_diverse_checker = true;
            } else if attr.path_matches(&[gpu_symbol(), nvvm_launch_bound()]) {
                let args = attr
                    .meta_item_list()
                    .unwrap()
                    .iter()
                    .map(|l| {
                        if let Some(lit) = l.lit() {
                            if let rustc_ast::LitKind::Int(value, int_ty) = lit.kind {
                                Some(value.get() as u32)
                            } else {
                                panic!("Expected an integer, found {:?} at {:?}", lit, attr.span());
                            }
                        } else if let Some(l) = l.value_str() {
                            if l.as_str() == "_" {
                                None
                            } else {
                                panic!(
                                    "Expected an integer or '_', found {:?} at {:?}",
                                    l,
                                    attr.span()
                                );
                            }
                        } else {
                            unreachable!()
                        }
                    })
                    .collect::<Vec<_>>();
                gpu_attrs.launch_bound = Some(NvmmLaunchBound {
                    max_thread_per_block: [
                        args[0].unwrap_or(1024) as i32,
                        args.get(1).unwrap_or(&None).unwrap_or(1024) as i32,
                        args.get(2).unwrap_or(&None).unwrap_or(1024) as i32,
                    ],
                    min_block_per_sm: *args.get(3).unwrap_or(&None),
                });
            } else if attr.path_matches(&[gpu_symbol()]) {
                panic!("Unknown gpu attribute: {:?}", attr);
            }
        }
        gpu_attrs
    }
}
