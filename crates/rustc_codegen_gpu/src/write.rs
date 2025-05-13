use std::process::Command;

use melior::ir::operation::OperationPrintingFlags;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::{
    back::write::{CodegenContext, ModuleConfig},
    CompiledModule,
};
use rustc_errors::DiagCtxtHandle;

use crate::attr::is_gpu_code;
use crate::{
    backend::{GPUCodeGenModule, GPUCodegenBackend},
    builder::GpuBuilder,
    context::GPUCodegenContext,
};
use rustc_span::Symbol;

pub(crate) fn codegen(
    cgcx: &CodegenContext<GPUCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<GPUCodeGenModule>,
    _config: &ModuleConfig,
) -> Result<rustc_codegen_ssa::CompiledModule, rustc_errors::FatalError> {
    let mod_name = module.name.clone();
    let module_name = Some(&mod_name[..]);
    log::trace!("try write MLIR module to");
    let out = if let Some(m) = module.module_llvm.mlir_module {
        let out = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Mir, module_name);
        let out_opt = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Assembly, module_name);
        let out_ll = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::LlvmAssembly, module_name);
        let out_bc = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Bitcode, module_name);
        let out_obj = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Object, module_name);
        let copy = format!("{}-copy", mod_name);
        let out_obj_copy = cgcx.output_filenames.temp_path(
            rustc_session::config::OutputType::Object,
            Some(copy.as_str()),
        );
        log::trace!("write MLIR module to {:?}", out);
        let content = m
            .module
            .as_operation()
            .to_string_with_flags(
                OperationPrintingFlags::new()
                    .use_local_scope()
                    .print_generic_operation_form(),
            )
            .unwrap();
        std::fs::write(&out, &content).unwrap();
        if !m.module.as_operation().verify() {
            log::trace!("MLIR module verify failed: {}", content);
            Err(rustc_errors::FatalError)?;
        }
        // mlir-opt must use "shell" in order to pass correct arguments.
        let mut mlir_opt = Command::new("sh");
        let cmd = format!(
            "{} {} -o {} {}",
            which::which("mlir-opt")
                .expect("mlir-opt not found")
                .display(),
            r#"-gpu-lower-to-nvvm-pipeline='opt-level=3 cubin-chip=sm_90a cubin-features=+ptx80'"#,
            out_opt.to_str().unwrap(),
            out.to_str().unwrap()
        );
        let args = ["-c", cmd.as_str()];
        let mlir_opt = mlir_opt
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());
        println!("{:?}", mlir_opt.get_args());
        let output = mlir_opt
            .spawn()
            .unwrap_or_else(|_| panic!("Failed to spawn mlir-opt"))
            .wait_with_output()
            .unwrap();
        if !output.status.success() {
            panic!("mlir-opt failed with status: {:?}", output.status);
        }

        let args = [
            "-mlir-to-llvmir",
            "-o",
            out_ll.to_str().unwrap(),
            out_opt.to_str().unwrap(),
        ];
        let mlir_translate = Command::new("mlir-translate")
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .unwrap();
        let output = mlir_translate.wait_with_output().unwrap();
        if !output.status.success() {
            panic!("mlir-translate failed with status: {:?}", output.status);
        }

        let args = ["-o", out_bc.to_str().unwrap(), out_ll.to_str().unwrap()];
        let mlir_translate = Command::new("llvm-as")
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .unwrap();
        let output = mlir_translate.wait_with_output().unwrap();
        if !output.status.success() {
            panic!("llvm-as failed with status: {:?}", output.status);
        }
        let status = Command::new("llc")
            .arg("-filetype=obj")
            .arg(out_bc.to_str().unwrap())
            .arg("-o")
            .arg(out_obj.to_str().unwrap())
            .status()
            .map_err(|e| format!("Failed to execute llc: {}", e))
            .unwrap();
        if !status.success() {
            panic!("llc failed with status: {:?}", output.status);
        }
        log::trace!("write MLIR obj to {:?}", out_obj);
        std::fs::copy(out_obj.to_str().unwrap(), out_obj_copy.to_str().unwrap()).unwrap();
        Some(out_obj_copy)
    } else {
        None
    };
    Ok(CompiledModule {
        name: mod_name,
        kind: module.kind,
        object: out,
        dwarf_object: None,
        bytecode: None,
        assembly: None,
        llvm_ir: None,
        links_from_incr_cache: vec![],
    })
}

pub(crate) fn module_codegen<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    cgu_name: Symbol,
) -> ModuleCodegen<GPUCodeGenModule> {
    let mlir_ctx = crate::mlir::create_mlir_ctx();
    let (mlir_module, gpu_block, cpu_block) = crate::mlir::create_top_module(mlir_ctx);
    let mut blocks = std::collections::HashMap::new();
    blocks.insert("host".to_string(), mlir_module.body());
    blocks.insert("gpu".to_string(), gpu_block);

    log::trace!("create MLIR module {}", cgu_name);
    let cgu = tcx.codegen_unit(cgu_name);
    {
        let cx: GPUCodegenContext<'_, '_, '_> =
            crate::context::GPUCodegenContext::new(tcx, mlir_ctx, &mlir_module, blocks);
        let mono_items = cgu.items_in_deterministic_order(tcx);
        for &(mono_item, data) in &mono_items {
            if is_gpu_code(&tcx, mono_item.def_id()) {
                log::trace!("predefine {}", mono_item);
            } else {
                log::trace!("skip {}", mono_item);
                continue;
            }
            mono_item.predefine::<GpuBuilder<'_, '_, '_>>(&cx, data.linkage, data.visibility);
        }
        for (mono_item, mono_data) in mono_items {
            if is_gpu_code(&tcx, mono_item.def_id()) {
                log::trace!("define {}", mono_item);
            } else {
                log::trace!("skip {}", mono_item);
                continue;
            }
            mono_item.define::<GpuBuilder<'_, '_, '_>>(&cx);
        }
    }

    let module = GPUCodeGenModule {
        llvm_module: None,
        mlir_module: Some(crate::backend::MLIRModule {
            module: mlir_module,
        }),
    };
    let m = ModuleCodegen::new_regular(cgu_name.to_string(), module);
    eprintln!("compile_codegen_unit {}", cgu_name);
    m
}
