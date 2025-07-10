use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

use tracing::info;

pub struct CompileConfig {
    pub opt_level: u8,
    pub cubin_chip: String,
    pub cubin_features: String,
    pub gpu_sym: String,
}

impl Default for CompileConfig {
    fn default() -> Self {
        CompileConfig {
            opt_level: 3,
            cubin_chip: "sm_80".to_string(),
            cubin_features: "+ptx80".to_string(),
            gpu_sym: "gpu_bin_cst".to_string(),
        }
    }
}

pub fn command<I, S>(bin: &str, args: I) -> std::io::Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());
    let output =
        cmd.spawn().unwrap_or_else(|_| panic!("Failed to spawn {}", bin)).wait_with_output()?;
    if !output.status.success() {
        panic!("{} failed with status: {:?}", bin, output.status);
    }
    Ok(())
}

impl CompileConfig {
    pub fn mlir_opt(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        // mlir-opt must use "shell" in order to pass correct arguments.
        info!("[mlir-opt] outputs {}", outpath.display());
        //--pass-pipeline='builtin.module(gpu-kernel-outlining,convert-gpu-to-nvvm{has-redux=1},convert-nvvm-to-llvm,reconcile-unrealized-casts)'

        //format!("-gpu-lower-to-nvvm-pipeline='opt-level={} cubin-chip={} cubin-features={}'",self.opt_level, self.cubin_chip, self.cubin_features)
        // mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp
        let mlir_opt_args = format!(
            "--pass-pipeline=\
            'builtin.module(\
            convert-nvgpu-to-nvvm,\
            gpu-kernel-outlining,\
            convert-vector-to-scf,\
            convert-scf-to-cf,\
            convert-nvvm-to-llvm,\
            convert-func-to-llvm,\
            expand-strided-metadata,\
            nvvm-attach-target{{triple=nvptx64-nvidia-cuda chip={} features={} O={}}},\
            lower-affine,\
            convert-arith-to-llvm,\
            convert-index-to-llvm{{index-bitwidth=64}},\
            canonicalize,\
            cse,\
            reconcile-unrealized-casts,\
            gpu.module(\
                convert-gpu-to-nvvm{{has-redux=1 use-bare-ptr-memref-call-conv=1}},\
                canonicalize,\
                cse),\
            gpu-to-llvm,\
            gpu-module-to-binary,convert-math-to-llvm,\
            reconcile-unrealized-casts, canonicalize,cse)'",
            self.cubin_chip, self.cubin_features, self.opt_level
        );

        let cmd = format!(
            "{} {} {} -o {} ",
            which::which("mlir-opt").expect("mlir-opt not found").display(),
            mlir_opt_args,
            inpath.to_str().unwrap(),
            outpath.to_str().unwrap(),
        );
        tracing::warn!("Running command: {}", cmd);
        let args = ["-c", cmd.as_str()];
        command("sh", args)
    }

    pub fn mlir_translate(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        info!("[mlir-translate] outputs {}", outpath.display());
        let args = ["-mlir-to-llvmir", inpath.to_str().unwrap(), "-o", outpath.to_str().unwrap()];
        command("mlir-translate", args)
    }

    pub fn llvm_as(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        let args = [inpath.to_str().unwrap(), "-o", outpath.to_str().unwrap()];
        command("llvm-as", args)
    }

    pub fn llc(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        let args = ["-filetype=obj", inpath.to_str().unwrap(), "-o", outpath.to_str().unwrap()];
        command("llc", args)
    }

    pub fn objcopy(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        info!("[objcopy] outputs {}", outpath.display());
        let flag = format!("--globalize-symbol={}", self.gpu_sym);
        let args = [flag.as_str(), inpath.to_str().unwrap(), outpath.to_str().unwrap()];
        command("objcopy", args)
    }

    pub fn mlir_compile(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        let opt_out = inpath.with_extension("opt.mlir.mir");
        self.mlir_opt(inpath, &opt_out)?;
        let llvmir_out = inpath.with_extension("ll");
        self.mlir_translate(&opt_out, &llvmir_out)?;
        let bitcode_out = inpath.with_extension("bc");
        self.llvm_as(&llvmir_out, &bitcode_out)?;
        let obj_out = inpath.with_extension("tmp.o");
        self.llc(&bitcode_out, &obj_out)?;
        let final_out = outpath.with_extension("o");
        self.objcopy(&obj_out, &final_out)?;
        Ok(())
    }
}
