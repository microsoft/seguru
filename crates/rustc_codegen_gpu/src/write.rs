use std::process::Command;

use rustc_codegen_ssa_gpu::back::write::{CodegenContext, ModuleConfig};
use rustc_codegen_ssa_gpu::{CompiledModule, ModuleCodegen};
use rustc_errors::DiagCtxtHandle;

use crate::backend::{GPUCodeGenModule, GPUCodegenBackend};

pub(crate) fn mlir_opt(inpath: &str, outpath: &str) -> Result<(), rustc_errors::FatalError> {
    // mlir-opt must use "shell" in order to pass correct arguments.
    let mut mlir_opt = Command::new("sh");
    let cmd = format!(
        "{} {} {} -o {} ",
        which::which("mlir-opt").expect("mlir-opt not found").display(),
        r#"-gpu-lower-to-nvvm-pipeline='opt-level=3 cubin-chip=sm_80 cubin-features=+ptx80'"#,
        inpath,
        outpath,
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
    Ok(())
}
pub(crate) fn codegen(
    cgcx: &CodegenContext<GPUCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<GPUCodeGenModule>,
    _config: &ModuleConfig,
) -> Result<rustc_codegen_ssa_gpu::CompiledModule, rustc_errors::FatalError> {
    let mod_name = module.name.clone();
    let module_name = Some(&mod_name[..]);
    let out = if let Some(m) = module.module_llvm.mlir_module {
        let out =
            cgcx.output_filenames.temp_path(rustc_session::config::OutputType::Mir, module_name);
        let out_opt = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Assembly, module_name);
        let out_ll = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::LlvmAssembly, module_name);
        let out_bc = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Bitcode, module_name);
        let out_obj =
            cgcx.output_filenames.temp_path(rustc_session::config::OutputType::Object, module_name);
        let copy = format!("{}-copy", mod_name);
        let out_obj_private = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Object, Some(copy.as_str()));
        log::debug!("write MLIR module to {:?}", out);

        if !m.module.as_operation().verify() {
            log::trace!("MLIR module verify failed.");
            Err(rustc_errors::FatalError)?;
        }
        let content = m.module.as_operation().to_string();
        let content = content.replace("attributes {kernel, ", "kernel attributes {");

        std::fs::write(&out, &content).unwrap();
        log::debug!("[Done]write MLIR module to {:?}", out);
        // mlir-opt must use "shell" in order to pass correct arguments.
        mlir_opt(out.to_str().unwrap(), out_opt.to_str().unwrap())?;

        let args = ["-mlir-to-llvmir", "-o", out_ll.to_str().unwrap(), out_opt.to_str().unwrap()];
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
            .arg(out_obj_private.to_str().unwrap())
            .status()
            .map_err(|e| format!("Failed to execute llc: {}", e))
            .unwrap();
        if !status.success() {
            panic!("llc failed with status: {:?}", output.status);
        }
        let status = Command::new("objcopy")
            .arg("--globalize-symbol=gpu_bin_cst")
            .arg(out_obj_private.to_str().unwrap())
            .arg(out_obj.to_str().unwrap())
            .status()
            .map_err(|e| format!("Failed to execute llc: {}", e))
            .unwrap();
        if !status.success() {
            panic!("objcopy failed with status: {:?}", output.status);
        }
        log::trace!("copy MLIR obj to {:?}", out_obj);
        Some(out_obj_private)
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
