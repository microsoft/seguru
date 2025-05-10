mod abi;
mod asm;
mod const_static;
mod coverage;
mod debug;
mod misc;
mod predef;
mod to_mir_func;
mod ty;

use rustc_abi::{self, HasDataLayout};
use rustc_codegen_ssa::traits::BackendTypes;
use std::marker::PhantomData;
use std::{collections::HashMap, sync::RwLock};

use melior::ir as mlir_ir;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv};

use crate::mlir::BlockRefWithTime;

use self::ty::MLIRType;

pub(crate) struct GPUCodegenContext<'tcx, 'ml, 'a> {
    pub mlir_ctx: &'ml melior::Context,
    pub mlir_module: &'ml melior::ir::Module<'ml>,
    pub mlir_body: melior::ir::BlockRef<'ml, 'a>,
    pub dummy: PhantomData<&'a mlir_ir::operation::Operation<'ml>>,
    pub fn_db: HashMap<rustc_hir::def_id::DefId, mlir_ir::operation::OperationRef<'ml, 'a>>,
    pub const_alloc: RwLock<HashMap<rustc_const_eval::interpret::AllocId, mlir_ir::Value<'ml, 'a>>>,
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
}

impl<'tcx, 'ml, 'a> std::fmt::Debug for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUCodegenContext")
            .field("mlir_ctx", &self.mlir_ctx)
            .field("mlir_module", &self.mlir_module)
            .field("mlir_body", &self.mlir_body)
            .field("dummy", &self.dummy)
            .finish()
    }
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub fn new(
        tcx: rustc_middle::ty::TyCtxt<'tcx>,
        mlir_ctx: &'ml melior::Context,
        mlir_module: &'ml melior::ir::Module<'ml>,
        mlir_body: melior::ir::BlockRef<'ml, 'ml>,
    ) -> Self {
        let location = melior::ir::Location::unknown(mlir_ctx);
        Self {
            mlir_ctx,
            tcx,
            mlir_module,
            mlir_body,
            dummy: PhantomData,
            fn_db: HashMap::new(),
            const_alloc: RwLock::new(HashMap::new()),
        }
    }

    pub fn mlir_body(&self) -> &'a melior::ir::Block<'ml> {
        self.mlir_body.to_ref()
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
