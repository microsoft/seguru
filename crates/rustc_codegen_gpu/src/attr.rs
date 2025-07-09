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
    pub shared_size: bool, // this is used to decide whether to call `gpu_shared_size`.
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum GpuItem {
    AllReduce,
    ThreadId,
    GlobalThreadId,
    BlockId,
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
}

impl TryFrom<&str> for GpuItem {
    type Error = ();

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let ret = match s {
            "gpu::all_reduce" => GpuItem::AllReduce,
            "gpu::thread_id" => GpuItem::ThreadId,
            "gpu::global_thread_id" => GpuItem::GlobalThreadId,
            "gpu::block_id" => GpuItem::BlockId,
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
            _ => return Err(()),
        };
        Ok(ret)
    }
}

impl From<GpuItem> for &'static str {
    fn from(item: GpuItem) -> &'static str {
        match item {
            GpuItem::AllReduce => "gpu::all_reduce",
            GpuItem::ThreadId => "gpu::thread_id",
            GpuItem::GlobalThreadId => "gpu::global_thread_id",
            GpuItem::BlockId => "gpu::block_id",
            GpuItem::LaneId => "gpu::lane_id",
            GpuItem::SubgroupId => "gpu::subgroup_id",
            GpuItem::SubgroupReduce => "gpu::subgroup_reduce",
            GpuItem::SubgroupSize => "gpu::subgroup_size",
            GpuItem::Shuffle => "gpu::shuffle",
            GpuItem::NvvmReduxSync => "nvvm::redux_sync",
            GpuItem::BlockDim => "gpu::block_dim",
            GpuItem::BlockId => "gpu::block_id",
            GpuItem::GridDim => "gpu::grid_dim",
            GpuItem::PrintFormat => "gpu::printf",
            GpuItem::PrintArgs => "gpu::print_args",
            GpuItem::AddStringAttr => "gpu::add_mlir_string_attr",
            GpuItem::Scope => "gpu::scope",
            GpuItem::Launch => "gpu::launch",
            GpuItem::NewChunk => "gpu::GpuChunksMut::new",
            GpuItem::UniqueChunk => "gpu::GpuChunksMut::unique_chunk",
            GpuItem::SyncThreads => "gpu::sync_threads",
            GpuItem::Subslice => "gpu::subslice",
            GpuItem::SubsliceMut => "gpu::subslice_mut",
            GpuItem::NewSharedMem => "gpu::new_shared_mem",
            GpuItem::AtomicAdd => "gpu::atomic_add",
        }
    }
}

impl From<GpuItem> for Symbol {
    fn from(item: GpuItem) -> Self {
        Symbol::intern(item.into())
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

impl GpuAttributes {
    pub fn build(tcx: &rustc_middle::ty::TyCtxt<'_>, def_id: DefId) -> GpuAttributes {
        let attrs = tcx.get_attrs_unchecked(def_id);
        let mut gpu_attr = GpuAttributes::parse(attrs);
        if let Some(sym) = tcx.all_diagnostic_items(()).id_to_name.get(&def_id) {
            if let Ok(gpu_item) = GpuItem::try_from(*sym) {
                gpu_attr.gpu_item = Some(gpu_item);
            }
        }

        gpu_attr
    }

    pub fn to_mlir_attribute<'ml>(
        &self,
        ctx: &'ml melior::Context,
    ) -> Option<melior::ir::Attribute<'ml>> {
        if self.is_builtin() {
            self.gpu_item.map(|gpu_item| {
                melior::ir::attribute::StringAttribute::new(ctx, gpu_item.into()).into()
            })
        } else {
            None
        }
    }

    pub fn is_gpu_related(&self) -> bool {
        self.kernel || self.host || self.device || self.gpu_item.is_some() || self.shared_size
    }

    pub fn is_builtin(&self) -> bool {
        match self.gpu_item {
            Some(GpuItem::NewChunk) => false,
            Some(_) => true,
            _ => false,
        }
    }

    fn parse(attrs: &[Attribute]) -> Self {
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
            if attr.path_matches(&[gpu_symbol(), gpu_shared_size_symbol()]) {
                gpu_attrs.shared_size = true;
            }
        }
        gpu_attrs
    }
}
