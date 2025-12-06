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
use crate::write::get_compile_config;

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
}

impl WriteBackendMethods for GPUCodegenBackend {
    type Module = GPUCodeGenModule;

    type TargetMachine = ();

    type TargetMachineError = u32;

    type ModuleBuffer = GpuModuleBuffer;

    type ThinData = ();

    type ThinBuffer = GpuModuleBuffer;

    fn run_thin_lto(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[std::path::PathBuf],
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(
            rustc_codegen_ssa_gpu::back::lto::SerializedModule<Self::ModuleBuffer>,
            WorkProduct,
        )>,
    ) -> (Vec<rustc_codegen_ssa_gpu::back::lto::ThinModule<Self>>, Vec<WorkProduct>) {
        (vec![], vec![])
    }

    fn print_pass_timings(&self) {
        print!("no timing");
    }

    fn print_statistics(&self) {
        print!("no statistics");
    }

    fn optimize(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        dcx: rustc_errors::DiagCtxtHandle<'_>,
        module: &mut rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
        config: &rustc_codegen_ssa_gpu::back::write::ModuleConfig,
    ) {
    }

    fn optimize_thin(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        thin_module: rustc_codegen_ssa_gpu::back::lto::ThinModule<Self>,
    ) -> rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module> {
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
        module
    }

    fn codegen(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        module: rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
        config: &rustc_codegen_ssa_gpu::back::write::ModuleConfig,
    ) -> rustc_codegen_ssa_gpu::CompiledModule {
        crate::write::codegen(cgcx, module, config).expect("codegen success")
    }

    fn prepare_thin(
        module: rustc_codegen_ssa_gpu::ModuleCodegen<Self::Module>,
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

    fn run_and_optimize_fat_lto(
        cgcx: &rustc_codegen_ssa_gpu::back::write::CodegenContext<Self>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[std::path::PathBuf],
        modules: Vec<rustc_codegen_ssa_gpu::back::write::FatLtoInput<Self>>,
    ) -> ModuleCodegen<Self::Module> {
        todo!();
    }
}

impl ExtraBackendMethods for GPUCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        methods: &[rustc_ast::expand::allocator::AllocatorMethod],
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

    fn name(&self) -> &'static str {
        "GPUCodegenBackend"
    }

    fn locale_resource(&self) -> &'static str {
        // Provide a dummy implementation or actual logic
        ""
    }

    fn init(&self, sess: &Session) {}

    fn provide(&self, providers: &mut Providers) {}

    fn target_config(&self, sess: &Session) -> rustc_codegen_ssa_gpu::TargetConfig {
        let target: String = sess.target.arch.clone().into_owned();
        let faked_features = match target.as_str() {
            "x86" | "x86_64" => {
                vec![Symbol::intern("sse"), Symbol::intern("sse2"), Symbol::intern("x87")]
            }
            "aarch64" => {
                vec![Symbol::intern("neon")]
            }
            _ => {
                vec![]
            }
        };
        rustc_codegen_ssa_gpu::TargetConfig {
            target_features: faked_features.clone(),
            unstable_target_features: faked_features,
            has_reliable_f16: true,
            has_reliable_f16_math: true,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>) -> Box<dyn std::any::Any> {
        // Provide a dummy implementation or actual logic
        Box::new(rustc_codegen_ssa_gpu::base::codegen_crate(
            GPUCodegenBackend::new(),
            tcx,
            tcx.sess.opts.cg.target_cpu.clone().unwrap_or_else(|| tcx.sess.target.cpu.to_string()),
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &rustc_session::Session,
        outputs: &rustc_session::config::OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        let (mut codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa_gpu::back::write::OngoingCodegen<Self>>()
            .expect("Expected OngoingCodegen, found Box<Any>")
            .join(sess);
        let mut bytecode_paths = vec![];
        for module in &codegen_results.modules {
            let Some(object_path) = &module.object else { continue };
            let bytecode_path = object_path.with_extension("gpu.bc");
            if bytecode_path.exists() {
                bytecode_paths.push(bytecode_path.clone());
            }
        }
        let mlir_compile_config = get_compile_config(&sess.opts.cg);
        let bc_file = outputs.with_extension("gpu.bc");
        let bc_lib_file = bc_file
            .parent()
            .unwrap()
            .join(format!("lib{}", bc_file.file_name().unwrap().to_str().unwrap()));
        if !bytecode_paths.is_empty() {
            mlir_compile_config
                .llvm_link(&bytecode_paths, &bc_lib_file)
                .expect("failed to link bc");
        }
        // GPU code should always be treated as lib.
        codegen_results
            .crate_info
            .crate_types
            .iter_mut()
            .for_each(|t| *t = rustc_session::config::CrateType::Rlib);
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
