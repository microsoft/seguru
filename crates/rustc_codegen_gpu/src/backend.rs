use std::any::Any;
use std::sync::Arc;

use rustc_codegen_ssa_gpu::traits::{CodegenBackend, ExtraBackendMethods, WriteBackendMethods};
use rustc_codegen_ssa_gpu::{CodegenResults, ModuleCodegen};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_span::Symbol;

use crate::mlir::{create_mlir_ctx, new_empty_module};

#[derive(Clone)]
pub struct GPUCodegenBackend();
unsafe impl rustc_data_structures::sync::DynSync for GPUCodegenBackend {}
unsafe impl Send for GPUCodegenBackend {}
unsafe impl Sync for GPUCodegenBackend {}

impl GPUCodegenBackend {
    pub fn new() -> Self {
        Self()
    }
}

pub struct MLIRModule {
    pub module: melior::ir::Module<'static>,
}

unsafe impl Send for MLIRModule {}
unsafe impl Sync for MLIRModule {}

pub struct GPUCodeGenModule {
    pub mlir_module: Option<MLIRModule>,
}

impl GPUCodegenBackend {}

pub struct GpuModuleBuffer(Vec<u8>);

impl rustc_codegen_ssa_gpu::traits::ModuleBufferMethods for GpuModuleBuffer {
    fn data(&self) -> &[u8] {
        &self.0
    }
}

impl rustc_codegen_ssa_gpu::traits::ThinBufferMethods for GpuModuleBuffer {
    fn data(&self) -> &[u8] {
        &self.0
    }
    fn thin_link_data(&self) -> &[u8] {
        &[]
    }
}

impl WriteBackendMethods for GPUCodegenBackend {
    type Module = GPUCodeGenModule;

    type TargetMachine = ();

    type TargetMachineError = u32;

    type ModuleBuffer = GpuModuleBuffer;

    type ThinData = ();

    type ThinBuffer = GpuModuleBuffer;

    fn run_link(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        dcx: rustc_errors::DiagCtxtHandle<'_>,
        modules: Vec<rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>>,
    ) -> Result<rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>, rustc_errors::FatalError> {
        todo!();
    }

    fn run_fat_lto(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        modules: Vec<rustc_codegen_ssa_gpu::back::write::FatLtoInput<Self>>,
        cached_modules: Vec<(
            rustc_codegen_ssa_gpu::back::lto::SerializedModule<Self::ModuleBuffer>,
            WorkProduct,
        )>,
    ) -> Result<rustc_codegen_ssa_gpu::back::lto::LtoModuleCodegen<Self>, rustc_errors::FatalError>
    {
        todo!();
    }

    fn run_thin_lto(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(
            rustc_codegen_ssa_gpu::back::lto::SerializedModule<Self::ModuleBuffer>,
            WorkProduct,
        )>,
    ) -> Result<
        (Vec<rustc_codegen_ssa_gpu::back::lto::LtoModuleCodegen<Self>>, Vec<WorkProduct>),
        rustc_errors::FatalError,
    > {
        Ok((vec![], vec![]))
    }

    fn print_pass_timings(&self) {
        print!("no timing");
    }

    fn print_statistics(&self) {
        print!("no statistics");
    }

    unsafe fn optimize(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        dcx: rustc_errors::DiagCtxtHandle<'_>,
        module: &mut rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
        config: &rustc_codegen_ssa_gpu::back::write::ModuleConfig,
    ) -> Result<(), rustc_errors::FatalError> {
        eprintln!("optimize starts");
        Ok(())
    }

    fn optimize_fat(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        llmod: &mut rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
    ) -> Result<(), rustc_errors::FatalError> {
        eprintln!("optimize_fat starts");
        Ok(())
    }

    unsafe fn optimize_thin(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        thin_module: rustc_codegen_ssa_gpu::back::lto::ThinModule<Self>,
    ) -> Result<rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>, rustc_errors::FatalError> {
        let module_str = std::str::from_utf8(thin_module.data()).expect("valid utf8");
        let module = ModuleCodegen {
            module_llvm: GPUCodeGenModule {
                mlir_module: Some(MLIRModule {
                    module: melior::ir::Module::parse(create_mlir_ctx(), module_str)
                        .expect("valid module"),
                }),
            },
            name: thin_module.name().to_string(),
            kind: rustc_codegen_ssa_gpu::ModuleKind::Regular,
            thin_lto_buffer: Some(vec![]),
        };
        Ok(module)
    }

    unsafe fn codegen(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        dcx: rustc_errors::DiagCtxtHandle<'_>,
        module: rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
        config: &rustc_codegen_ssa_gpu::back::write::ModuleConfig,
    ) -> Result<rustc_codegen_ssa_gpu::CompiledModule, rustc_errors::FatalError> {
        eprintln!("codegen starts");
        let ret = crate::write::codegen(cgcx, dcx, module, config);
        eprintln!("codegen end");
        ret
    }

    fn prepare_thin(
        module: rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
        want_summary: bool,
    ) -> (String, Self::ThinBuffer) {
        Self::serialize_module(module)
    }

    fn serialize_module(
        module: rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
    ) -> (String, Self::ModuleBuffer) {
        let mut buffer = vec![];
        if let Some(module) = module.module_llvm.mlir_module {
            buffer = module.module.as_operation().to_string().into_bytes();
        }
        (module.name, GpuModuleBuffer(buffer))
    }

    fn autodiff(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        module: &rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
        diff_fncs: Vec<rustc_ast::expand::autodiff_attrs::AutoDiffItem>,
        config: &rustc_codegen_ssa_gpu::back::write::ModuleConfig,
    ) -> Result<(), rustc_errors::FatalError> {
        eprintln!("autodiff");
        Ok(())
    }
}

impl ExtraBackendMethods for GPUCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        kind: rustc_ast::expand::allocator::AllocatorKind,
        alloc_error_handler_kind: rustc_ast::expand::allocator::AllocatorKind,
    ) -> Self::Module {
        GPUCodeGenModule {
            mlir_module: Some(MLIRModule { module: new_empty_module(create_mlir_ctx()) }),
        }
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>, u64) {
        let start_time = std::time::Instant::now();
        let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
        let (module, _) = rustc_middle::ty::print::with_no_trimmed_paths!({
            tcx.dep_graph.with_task(
                dep_node,
                tcx,
                cgu_name,
                crate::codegen::module_codegen,
                Some(rustc_middle::dep_graph::hash_result),
            )
        });
        let time_to_codegen = start_time.elapsed();
        eprintln!("compile_codegen_unit {}", cgu_name);
        let cost = time_to_codegen.as_nanos() as u64;
        (module, cost)
    }

    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: rustc_session::config::OptLevel,
        target_features: &[String],
    ) -> rustc_codegen_ssa_gpu::back::write::TargetMachineFactoryFn<Self> {
        Arc::new(|_| Ok(()))
    }
}
impl CodegenBackend for GPUCodegenBackend {
    // Implement codegen methods

    fn locale_resource(&self) -> &'static str {
        // Provide a dummy implementation or actual logic
        ""
    }

    fn init(&self, sess: &Session) {}

    fn provide(&self, providers: &mut Providers) {}

    fn codegen_crate(
        &self,
        tcx: TyCtxt<'_>,
        metadata: rustc_metadata::EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn std::any::Any> {
        // Provide a dummy implementation or actual logic
        let x = Box::new(rustc_codegen_ssa_gpu::base::codegen_crate(
            GPUCodegenBackend::new(),
            tcx,
            tcx.sess.opts.cg.target_cpu.clone().unwrap_or_else(|| tcx.sess.target.cpu.to_string()),
            metadata,
            need_metadata_module,
        ));
        eprintln!("codegen_crate starts");
        x
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &rustc_session::Session,
        _outputs: &rustc_session::config::OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        eprintln!("join_codegen");
        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa_gpu::back::write::OngoingCodegen<Self>>()
            .expect("Expected OngoingCodegen, found Box<Any>")
            .join(sess);
        eprintln!("join_codegen");
        (codegen_results, work_products)
    }

    /*fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        outputs: &rustc_session::config::OutputFilenames,
    ) {
        //eprintln!("todo link {:?}", codegen_results.modules);
    }*/
}
