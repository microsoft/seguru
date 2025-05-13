use rustc_codegen_llvm::LlvmCodegenBackend;
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, ThinModule, ThinShared};
use rustc_codegen_ssa::traits::{CodegenBackend, ExtraBackendMethods, WriteBackendMethods};
use rustc_codegen_ssa::{CodegenResults, ModuleCodegen};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_span::Symbol;
use std::any::Any;
use std::sync::Arc;

type LlvmCodegenModule = <LlvmCodegenBackend as WriteBackendMethods>::Module;

#[derive(Clone)]
pub struct GPUCodegenBackend();
unsafe impl rustc_data_structures::sync::DynSync for GPUCodegenBackend {}
unsafe impl Send for GPUCodegenBackend {}
unsafe impl Sync for GPUCodegenBackend {}

pub fn llvm_backend() -> Box<LlvmCodegenBackend> {
    unsafe {
        let raw_ptr: *const dyn CodegenBackend = &*LlvmCodegenBackend::new();
        let casted_ptr: *const LlvmCodegenBackend = raw_ptr.cast();

        // We can safely create a new Box from the casted pointer
        let box_casted: Box<LlvmCodegenBackend> =
            Box::from_raw(casted_ptr as *mut LlvmCodegenBackend);
        box_casted
    }
}

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
    pub llvm_module: Option<LlvmCodegenModule>,
    pub mlir_module: Option<MLIRModule>,
}

impl GPUCodegenBackend {
    fn to_llvm_context(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
    ) -> rustc_codegen_ssa::back::write::CodegenContext<LlvmCodegenBackend> {
        rustc_codegen_ssa::back::write::CodegenContext::<LlvmCodegenBackend> {
            prof: cgcx.prof.clone(),
            lto: cgcx.lto.clone(),
            save_temps: cgcx.save_temps,
            fewer_names: cgcx.fewer_names,
            time_trace: cgcx.time_trace,
            exported_symbols: cgcx.exported_symbols.clone(),
            opts: cgcx.opts.clone(),
            crate_types: cgcx.crate_types.clone(),
            each_linked_rlib_for_lto: cgcx.each_linked_rlib_for_lto.clone(),
            output_filenames: cgcx.output_filenames.clone(),
            regular_module_config: cgcx.regular_module_config.clone(),
            metadata_module_config: cgcx.metadata_module_config.clone(),
            allocator_module_config: cgcx.allocator_module_config.clone(),
            tm_factory: cgcx.tm_factory.clone(),
            msvc_imps_needed: cgcx.msvc_imps_needed,
            is_pe_coff: cgcx.is_pe_coff,
            target_can_use_split_dwarf: cgcx.target_can_use_split_dwarf,
            target_arch: cgcx.target_arch.clone(),
            target_is_like_osx: cgcx.target_is_like_osx,
            target_is_like_aix: cgcx.target_is_like_aix,
            split_debuginfo: cgcx.split_debuginfo,
            split_dwarf_kind: cgcx.split_dwarf_kind,
            expanded_args: cgcx.expanded_args.clone(),
            diag_emitter: cgcx.diag_emitter.clone(),
            remark: cgcx.remark.clone(),
            remark_dir: cgcx.remark_dir.clone(),
            incr_comp_session_dir: cgcx.incr_comp_session_dir.clone(),
            coordinator_send: cgcx.coordinator_send.clone(),
            parallel: cgcx.parallel,
            pointer_size: cgcx.pointer_size,
        }
    }

    fn module_code_gen_from_llvm(
        m: rustc_codegen_ssa::ModuleCodegen<LlvmCodegenModule>,
    ) -> rustc_codegen_ssa::ModuleCodegen<GPUCodeGenModule> {
        ModuleCodegen {
            module_llvm: GPUCodeGenModule {
                llvm_module: Some(m.module_llvm),
                mlir_module: None,
            },
            name: m.name,
            kind: m.kind,
            thin_lto_buffer: m.thin_lto_buffer,
        }
    }

    fn to_llvm_modele_code_gen(
        m: rustc_codegen_ssa::ModuleCodegen<GPUCodeGenModule>,
    ) -> rustc_codegen_ssa::ModuleCodegen<LlvmCodegenModule> {
        ModuleCodegen::<LlvmCodegenModule> {
            module_llvm: m.module_llvm.llvm_module.unwrap(),
            name: m.name,
            kind: m.kind,
            thin_lto_buffer: m.thin_lto_buffer,
        }
    }

    fn thin_module_from_llvm(
        m: rustc_codegen_ssa::back::lto::ThinModule<LlvmCodegenBackend>,
    ) -> rustc_codegen_ssa::back::lto::ThinModule<GPUCodegenBackend> {
        let mm: ThinShared<LlvmCodegenBackend> =
            match Arc::<ThinShared<LlvmCodegenBackend>>::try_unwrap(m.shared) {
                Ok(mm) => mm,
                Err(_) => panic!("failed to unwrap ThinShared"),
            };

        ThinModule {
            shared: Arc::new(ThinShared {
                data: mm.data,
                thin_buffers: mm.thin_buffers,
                serialized_modules: mm.serialized_modules,
                module_names: mm.module_names,
            }),
            idx: m.idx,
        }
    }

    fn to_llvm_thin_module(
        m: rustc_codegen_ssa::back::lto::ThinModule<GPUCodegenBackend>,
    ) -> rustc_codegen_ssa::back::lto::ThinModule<LlvmCodegenBackend> {
        let mm: ThinShared<GPUCodegenBackend> =
            match Arc::<ThinShared<GPUCodegenBackend>>::try_unwrap(m.shared) {
                Ok(mm) => mm,
                Err(_) => panic!("failed to unwrap ThinShared"),
            };

        ThinModule {
            shared: Arc::new(ThinShared {
                data: mm.data,
                thin_buffers: mm.thin_buffers,
                serialized_modules: mm.serialized_modules,
                module_names: mm.module_names,
            }),
            idx: m.idx,
        }
    }

    fn lto_module_from_llvm(m: LtoModuleCodegen<LlvmCodegenBackend>) -> LtoModuleCodegen<Self> {
        match m {
            LtoModuleCodegen::Fat(m) => LtoModuleCodegen::Fat(Self::module_code_gen_from_llvm(m)),
            LtoModuleCodegen::Thin(m) => LtoModuleCodegen::Thin(Self::thin_module_from_llvm(m)),
        }
    }
}

impl WriteBackendMethods for GPUCodegenBackend {
    type Module = GPUCodeGenModule;

    type TargetMachine = <LlvmCodegenBackend as WriteBackendMethods>::TargetMachine;

    type TargetMachineError = <LlvmCodegenBackend as WriteBackendMethods>::TargetMachineError;

    type ModuleBuffer = <LlvmCodegenBackend as WriteBackendMethods>::ModuleBuffer;

    type ThinData = <LlvmCodegenBackend as WriteBackendMethods>::ThinData;

    type ThinBuffer = <LlvmCodegenBackend as WriteBackendMethods>::ThinBuffer;

    fn run_link(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        dcx: rustc_errors::DiagCtxtHandle<'_>,
        modules: Vec<rustc_codegen_ssa::ModuleCodegen<Self::Module>>,
    ) -> Result<rustc_codegen_ssa::ModuleCodegen<Self::Module>, rustc_errors::FatalError> {
        eprintln!("run_link starts");
        let cgcx: rustc_codegen_ssa::back::write::CodegenContext<_> =
            GPUCodegenBackend::to_llvm_context(cgcx);
        let ret = LlvmCodegenBackend::run_link(
            &cgcx,
            dcx,
            modules
                .into_iter()
                .map(Self::to_llvm_modele_code_gen)
                .collect(),
        );
        ret.map(Self::module_code_gen_from_llvm)
    }

    fn run_fat_lto(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        modules: Vec<rustc_codegen_ssa::back::write::FatLtoInput<Self>>,
        cached_modules: Vec<(
            rustc_codegen_ssa::back::lto::SerializedModule<Self::ModuleBuffer>,
            WorkProduct,
        )>,
    ) -> Result<rustc_codegen_ssa::back::lto::LtoModuleCodegen<Self>, rustc_errors::FatalError>
    {
        todo!();
    }

    fn run_thin_lto(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(
            rustc_codegen_ssa::back::lto::SerializedModule<Self::ModuleBuffer>,
            WorkProduct,
        )>,
    ) -> Result<
        (
            Vec<rustc_codegen_ssa::back::lto::LtoModuleCodegen<Self>>,
            Vec<WorkProduct>,
        ),
        rustc_errors::FatalError,
    > {
        eprintln!("run_thin_lto starts");
        let cgcx = GPUCodegenBackend::to_llvm_context(cgcx);
        LlvmCodegenBackend::run_thin_lto(&cgcx, modules, cached_modules).map(
            |(modules, work_products)| {
                (
                    modules
                        .into_iter()
                        .map(Self::lto_module_from_llvm)
                        .collect(),
                    work_products,
                )
            },
        )
    }

    fn print_pass_timings(&self) {
        print!("no timing");
    }

    fn print_statistics(&self) {
        print!("no statistics");
    }

    unsafe fn optimize(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        dcx: rustc_errors::DiagCtxtHandle<'_>,
        module: &mut rustc_codegen_ssa::ModuleCodegen<Self::Module>,
        config: &rustc_codegen_ssa::back::write::ModuleConfig,
    ) -> Result<(), rustc_errors::FatalError> {
        eprintln!("optimize starts");
        Ok(())
    }

    fn optimize_fat(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        llmod: &mut rustc_codegen_ssa::ModuleCodegen<Self::Module>,
    ) -> Result<(), rustc_errors::FatalError> {
        eprintln!("optimize_fat starts");
        Ok(())
    }

    unsafe fn optimize_thin(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        thin: rustc_codegen_ssa::back::lto::ThinModule<Self>,
    ) -> Result<rustc_codegen_ssa::ModuleCodegen<Self::Module>, rustc_errors::FatalError> {
        eprintln!("optimize_thin starts");
        LlvmCodegenBackend::optimize_thin(
            &GPUCodegenBackend::to_llvm_context(cgcx),
            GPUCodegenBackend::to_llvm_thin_module(thin),
        )
        .map(GPUCodegenBackend::module_code_gen_from_llvm)
    }

    unsafe fn codegen(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        dcx: rustc_errors::DiagCtxtHandle<'_>,
        module: rustc_codegen_ssa::ModuleCodegen<Self::Module>,
        config: &rustc_codegen_ssa::back::write::ModuleConfig,
    ) -> Result<rustc_codegen_ssa::CompiledModule, rustc_errors::FatalError> {
        eprintln!("codegen starts");
        let ret = crate::write::codegen(cgcx, dcx, module, config);
        eprintln!("codegen end");
        ret
    }

    fn prepare_thin(
        module: rustc_codegen_ssa::ModuleCodegen<Self::Module>,
        want_summary: bool,
    ) -> (String, Self::ThinBuffer) {
        eprintln!("prepare_thin starts");
        LlvmCodegenBackend::prepare_thin(Self::to_llvm_modele_code_gen(module), want_summary)
    }

    fn serialize_module(
        module: rustc_codegen_ssa::ModuleCodegen<Self::Module>,
    ) -> (String, Self::ModuleBuffer) {
        eprintln!("serialize_module starts");
        LlvmCodegenBackend::serialize_module(Self::to_llvm_modele_code_gen(module))
    }

    fn autodiff(
        cgcx: &rustc_codegen_ssa::back::write::CodegenContext<Self>,
        module: &rustc_codegen_ssa::ModuleCodegen<Self::Module>,
        diff_fncs: Vec<rustc_ast::expand::autodiff_attrs::AutoDiffItem>,
        config: &rustc_codegen_ssa::back::write::ModuleConfig,
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
        eprintln!("codegen_allocator");
        let llvm_mod =
            llvm_backend().codegen_allocator(tcx, module_name, kind, alloc_error_handler_kind);
        GPUCodeGenModule {
            llvm_module: Some(llvm_mod),
            mlir_module: None,
        }
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (rustc_codegen_ssa::ModuleCodegen<Self::Module>, u64) {
        let start_time = std::time::Instant::now();
        //let (llvm_module, cost) = llvm_backend().compile_codegen_unit(tcx, cgu_name);
        let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
        let (module, _) = tcx.dep_graph.with_task(
            dep_node,
            tcx,
            cgu_name,
            crate::write::module_codegen,
            Some(rustc_middle::dep_graph::hash_result),
        );
        let time_to_codegen = start_time.elapsed();
        eprintln!("compile_codegen_unit {}", cgu_name);
        let cost = time_to_codegen.as_nanos() as u64;
        //module.module_llvm.llvm_module = Some(llvm_module.module_llvm);
        (module, cost)
    }

    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: rustc_session::config::OptLevel,
        target_features: &[String],
    ) -> rustc_codegen_ssa::back::write::TargetMachineFactoryFn<Self> {
        eprintln!("target_machine_factory");
        llvm_backend().target_machine_factory(sess, opt_level, target_features)
    }
}
impl CodegenBackend for GPUCodegenBackend {
    // Implement codegen methods

    fn locale_resource(&self) -> &'static str {
        // Provide a dummy implementation or actual logic
        ""
    }

    fn init(&self, sess: &Session) {
        llvm_backend().init(sess); // Make sure llvm is inited
    }

    fn provide(&self, providers: &mut Providers) {
        llvm_backend().provide(providers);
    }

    fn codegen_crate(
        &self,
        tcx: TyCtxt<'_>,
        metadata: rustc_metadata::EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn std::any::Any> {
        // Provide a dummy implementation or actual logic
        let x = Box::new(rustc_codegen_ssa::base::codegen_crate(
            GPUCodegenBackend::new(),
            tcx,
            tcx.sess
                .opts
                .cg
                .target_cpu
                .clone()
                .unwrap_or_else(|| tcx.sess.target.cpu.to_string()),
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
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<Self>>()
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
