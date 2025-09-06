use rustc_hir::def_id::DefId;
use rustc_hir::{Attribute, LangItem};
use rustc_span::Symbol;

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
    AtomicAdd,
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
    tracing::warn!("lang_item_from_str: {}", name);
    LangItem::from_name(Symbol::intern(name))
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
            "gpu::atomic_add" => GpuItem::AtomicAdd,
            "gpu::build_sfi" => GpuItem::BuildSFI,
            "gpu::get_local_mut_2d" => GpuItem::GetLocalMut2D,
            "gpu::get_local_2d" => GpuItem::GetLocal2D,
            "gpu::base_dynamic_shared" => GpuItem::DynamicShared,
            s if s.starts_with("gpu::device_intrinsics::") => {
                GpuItem::DeviceIntrinsic(s.replace("gpu::device_intrinsics::", ""))
            }
            s if s.starts_with("core") => {
                if let Some(i) = lang_item_from_str(s) {
                    GpuItem::Core(i)
                } else {
                    GpuItem::CoreFn(s.to_string())
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
            GpuItem::AtomicAdd => "gpu::atomic_add".into(),
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

impl GpuAttributes {
    pub fn callee_device() -> Self {
        GpuAttributes { device: true, ..GpuAttributes::default() }
    }

    pub fn build(tcx: &rustc_middle::ty::TyCtxt<'_>, def_id: DefId) -> GpuAttributes {
        let attrs = tcx.get_attrs_unchecked(def_id);
        let mut gpu_attr = GpuAttributes::parse(attrs);
        if tcx.crate_name(def_id.krate).as_str() == "core" {
            if let Some(lang_item) = tcx.lang_items().from_def_id(def_id) {
                if matches!(lang_item, LangItem::PanicBoundsCheck | LangItem::PanicNounwind) {
                    gpu_attr.device = true;
                    gpu_attr.gpu_item = Some(GpuItem::Core(lang_item));
                }
            } else {
                let path = tcx.def_path_str(def_id);
                let path = path.as_str();
                if [
                    "core::slice::index::slice_index_order_fail",
                    "core::slice::index::slice_start_index_len_fail",
                    "core::slice::index::slice_end_index_len_fail",
                ]
                .contains(&path)
                {
                    gpu_attr.device = true;
                    gpu_attr.gpu_item = Some(GpuItem::CoreFn(tcx.def_path_str(def_id)));
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

    pub fn to_mlir_attribute<'ml>(
        &self,
        ctx: &'ml melior::Context,
    ) -> Option<melior::ir::Attribute<'ml>> {
        if self.is_builtin() {
            self.gpu_item.clone().map(|gpu_item| {
                melior::ir::attribute::StringAttribute::new(ctx, &String::from(gpu_item)).into()
            })
        } else {
            None
        }
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
            Some(GpuItem::Core(item))
                if !matches!(item, LangItem::PanicBoundsCheck | LangItem::PanicNounwind) =>
            {
                false
            }
            Some(GpuItem::CoreFn(path))
                if ![
                    "core::slice::index::slice_index_order_fail",
                    "core::slice::index::slice_start_index_len_fail",
                    "core::slice::index::slice_end_index_len_fail",
                ]
                .contains(&path.as_str()) =>
            {
                false
            }
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
            }
            if attr.path_matches(&[gpu_symbol(), host_symbol()]) {
                gpu_attrs.host = true;
            }
            if attr.path_matches(&[gpu_symbol(), device_symbol()]) {
                gpu_attrs.device = true;
            }
            if attr.path_matches(&[gpu_symbol(), gpu_shared_size_symbol()]) {
                gpu_attrs.shared_size = true;
            }
            if attr.path_matches(&[gpu_symbol(), memspace_shared()]) {
                let shared_data = get_meta_usize_list(attr);
                gpu_attrs.shared_data = shared_data;
            }

            if attr.path_matches(&[gpu_symbol(), sync_data()]) {
                let sync_data = get_meta_usize_list(attr);
                gpu_attrs.sync_data = Some(sync_data);
            }

            if attr.path_matches(&[gpu_symbol(), ret_sync_data()]) {
                let ret_sync_data = get_meta_usize_list(attr);
                gpu_attrs.ret_sync_data = ret_sync_data;
            }
        }
        gpu_attrs
    }
}
