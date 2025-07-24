mod abi;
mod asm;
pub mod const_static;
mod coverage;
mod debug;
mod misc;
mod predef;
pub(crate) mod to_mir_func;
mod ty;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::RwLock;
use std::sync::atomic::AtomicUsize;

use melior::ir::{self as mlir_ir, TypeLike};
use rustc_abi::{self, HasDataLayout};
use rustc_codegen_ssa_gpu::traits::BackendTypes;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv};

use self::ty::MLIRType;
use crate::mlir::BlockRefWithTime;

type CodegenGPUError = String;

pub(crate) struct GPUCodegenContext<'tcx, 'ml, 'a> {
    pub cgu_name: String,
    pub cgu: &'tcx rustc_middle::mir::mono::CodegenUnit<'tcx>,
    pub mlir_ctx: &'ml melior::Context,
    pub mlir_module: &'ml melior::ir::Module<'ml>,
    pub mlir_body: HashMap<String, melior::ir::BlockRef<'ml, 'ml>>,
    pub dummy: PhantomData<&'a mlir_ir::operation::Operation<'ml>>,
    pub fn_db: RwLock<HashMap<String, mlir_ir::operation::OperationRef<'ml, 'a>>>,
    pub fn_ptr_db:
        RwLock<HashMap<String, (rustc_middle::ty::Instance<'tcx>, mlir_ir::Value<'ml, 'a>)>>,
    pub indirect_entry:
        std::sync::Mutex<Option<crate::context::to_mir_func::IndirectEntry<'tcx, 'ml>>>,
    pub const_alloc: RwLock<HashMap<rustc_const_eval::interpret::AllocId, mlir_ir::Value<'ml, 'a>>>,
    pub const_name_to_allocid: RwLock<HashMap<String, rustc_const_eval::interpret::AllocId>>,
    pub span_to_types: RwLock<HashMap<rustc_span::Span, mlir_ir::Type<'ml>>>,
    pub fn_shared_memory_size: RwLock<HashMap<String, usize>>,
    pub expected_shared_memory_size: RwLock<HashMap<String, (rustc_span::Span, usize)>>,
    pub static_shared_count: AtomicUsize,
    pub tcx: rustc_middle::ty::TyCtxt<'tcx>,
}

impl<'tcx, 'ml, 'a> std::fmt::Debug for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUCodegenContext")
            .field("mlir_ctx", &self.mlir_ctx)
            .field("mlir_module", &self.mlir_module)
            .field("dummy", &self.dummy)
            .finish()
    }
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    #[inline(always)]
    pub fn emit_error(&self, msg: String, span: impl Into<rustc_errors::MultiSpan>) -> ! {
        self.tcx.sess.dcx().span_fatal(span, msg)
    }

    pub fn new(
        cgu_name: String,
        cgu: &'tcx rustc_middle::mir::mono::CodegenUnit<'tcx>,
        tcx: rustc_middle::ty::TyCtxt<'tcx>,
        mlir_ctx: &'ml melior::Context,
        mlir_module: &'ml melior::ir::Module<'ml>,
        mlir_body: HashMap<String, melior::ir::BlockRef<'ml, 'ml>>,
    ) -> Self {
        let location = melior::ir::Location::unknown(mlir_ctx);
        Self {
            cgu_name,
            cgu,
            mlir_ctx,
            tcx,
            mlir_module,
            mlir_body,
            dummy: PhantomData,
            fn_db: RwLock::new(HashMap::new()),
            fn_ptr_db: RwLock::new(HashMap::new()),
            indirect_entry: std::sync::Mutex::new(None),
            const_alloc: RwLock::new(HashMap::new()),
            const_name_to_allocid: RwLock::new(HashMap::new()),
            span_to_types: RwLock::new(HashMap::new()),
            fn_shared_memory_size: RwLock::new(HashMap::new()),
            expected_shared_memory_size: RwLock::new(HashMap::new()),
            static_shared_count: AtomicUsize::new(0),
        }
    }

    pub fn mlir_body(&self, gpu: bool) -> &'a melior::ir::Block<'ml> {
        let cpu_block = unsafe { self.mlir_body["host"].to_ref() };
        if !gpu {
            return cpu_block;
        }
        let gpu_block = unsafe { self.mlir_body["gpu"].to_ref() };
        gpu_block
    }

    pub fn get_const_bytes_by_name(&self, name: &str) -> &[u8] {
        let alloc_id = self.const_name_to_allocid.read().unwrap()[name];
        let alloc = self.tcx.global_alloc(alloc_id).unwrap_memory();
        let bytes = self.tcx.global_alloc(alloc_id).unwrap_memory().inner().get_bytes_unchecked(
            rustc_const_eval::interpret::alloc_range(rustc_abi::Size::ZERO, alloc.inner().size()),
        );
        bytes
    }
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub fn unknown_loc(&self) -> melior::ir::Location<'ml> {
        melior::ir::Location::unknown(self.mlir_ctx)
    }

    pub fn to_mlir_loc(&self, span: rustc_span::Span) -> melior::ir::Location<'ml> {
        let source_map = self.tcx.sess.source_map();
        let loc = source_map.lookup_char_pos(span.lo());

        melior::ir::Location::new(
            self.mlir_ctx,
            format!("{}", loc.file.name.prefer_local()).as_str(),
            loc.line,
            loc.col.0,
        )
    }

    pub fn use_raw_type(&self, ty: ty::MLIRType<'_>) -> ty::MLIRType<'ml> {
        unsafe { ty::MLIRType::from_raw(ty.to_raw()) }
    }
}

impl<'tcx, 'ml, 'a> HasTypingEnv<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn typing_env(&self) -> rustc_middle::ty::TypingEnv<'tcx> {
        rustc_middle::ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx, 'ml, 'a> HasTyCtxt<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn tcx(&self) -> rustc_middle::ty::TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx, 'ml, 'a> HasDataLayout for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.tcx.data_layout()
    }
}

impl<'tcx, 'ml, 'a> BackendTypes for GPUCodegenContext<'tcx, 'ml, 'a> {
    type Value = mlir_ir::Value<'ml, 'a>;

    type Metadata = ();

    type Function = mlir_ir::operation::OperationRef<'ml, 'a>;

    type BasicBlock = mlir_ir::block::BlockRef<'ml, 'a>;

    type Type = MLIRType<'ml>;

    /// Each Block may contain an instance of this, indicating whether the block is part of a landing pad or not. This is used to make
    /// decision about whether to emit invoke instructions (e.g., in a landing pad we don’t continue to use invoke) and also about
    /// various function call metadata.
    type Funclet = ();

    type DIScope = ();

    type DILocation = ();

    type DIVariable = ();
}
