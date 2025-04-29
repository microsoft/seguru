use rustc_codegen_llvm::LlvmCodegenBackend;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::CodegenResults;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::{
    intravisit::{walk_expr, Visitor},
    Expr, ExprKind,
};
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_span::Symbol;
use std::any::Any;
use std::sync::Arc;

#[derive(Clone)]
struct GPUCodegenBackend {
    llvm_backend: Arc<Box<dyn CodegenBackend>>,
}

struct ForLoopPrinter<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for ForLoopPrinter<'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(func_expr, _) = &expr.kind {
            eprintln!("Consider reviewing this call for translation");
            if let ExprKind::Path(rustc_hir::QPath::Resolved(None, fun_path)) = func_expr.kind {
                let Some(def_id) = fun_path.res.opt_def_id() else {
                    // 🎯 Got it!
                    panic!("Function call without def_id");
                };
                if self.tcx.def_path_str(def_id) == "gpu::scope" {
                    dbg!(expr);
                    // This is where we would translate the function
                    // to MLIR or LLVM IR.
                }
            }
        }
        walk_expr(self, expr);
    }
}

impl<'tcx> ForLoopPrinter<'tcx> {
    pub fn printall_loops(&mut self) {
        eprintln!("Printing all loops in the crate");
        for item_id in self.tcx.hir().items() {
            
            let item = self.tcx.hir().item(item_id);
            if let rustc_hir::ItemKind::Fn { body, .. } = &item.kind {
                dbg!(body);
                let body = self.tcx.hir().body(*body);
                self.visit_body(body);
            }
        }
    }
}

impl CodegenBackend for GPUCodegenBackend {
    // Implement codegen methods

    fn locale_resource(&self) -> &'static str {
        // Provide a dummy implementation or actual logic
        ""
    }

    fn init(&self, sess: &Session) {
        self.llvm_backend.init(sess); // Make sure llvm is inited
    }

    fn provide(&self, providers: &mut Providers) {
        self.llvm_backend.provide(providers);
    }

    fn codegen_crate(
        &self,
        tcx: TyCtxt<'_>,
        metadata: rustc_metadata::EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn std::any::Any> {
        // Provide a dummy implementation or actual logic
        let mut visitor = ForLoopPrinter { tcx };
        visitor.printall_loops();
        self.llvm_backend
            .codegen_crate(tcx, metadata, need_metadata_module)
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &rustc_session::Session,
        outputs: &rustc_session::config::OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        self.llvm_backend
            .join_codegen(ongoing_codegen, sess, outputs)
    }
}

#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(GPUCodegenBackend {
        llvm_backend: Arc::new(LlvmCodegenBackend::new()),
    })
}
