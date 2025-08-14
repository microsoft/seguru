use std::ffi::OsStr;
use std::path::{Path, PathBuf};
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

fn find_libdevice() -> Option<PathBuf> {
    // Try CUDA_HOME first
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let path = Path::new(&cuda_home).join("nvvm/libdevice/libdevice.10.bc");
        if path.exists() {
            return Some(path);
        }
    }

    // Try default installation path
    let default_path = Path::new("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc");
    if default_path.exists() {
        return Some(default_path.to_path_buf());
    }

    // Optionally: scan other possible versions
    let libdevice_dir = Path::new("/usr/local/cuda/nvvm/libdevice/");
    if let Ok(entries) = libdevice_dir.read_dir() {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path.file_name().unwrap_or_default().to_string_lossy().starts_with("libdevice")
                && path.extension().map(|s| s == "bc").unwrap_or(false)
            {
                return Some(path);
            }
        }
    }

    // Nothing found
    None
}

impl CompileConfig {
    pub fn mlir_opt(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        // mlir-opt must use "shell" in order to pass correct arguments.
        info!("[mlir-opt] outputs {}", outpath.display());
        //--pass-pipeline='builtin.module(gpu-kernel-outlining,convert-gpu-to-nvvm{has-redux=1},convert-nvvm-to-llvm,reconcile-unrealized-casts)'

        //format!("-gpu-lower-to-nvvm-pipeline='opt-level={} cubin-chip={} cubin-features={}'",self.opt_level, self.cubin_chip, self.cubin_features)
        // mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp
        let dev_lib = find_libdevice().expect("Cannot find libdevice.10.bc");
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
            mem2reg,\
            nvvm-attach-target{{triple=nvptx64-nvidia-cuda fast ftz chip={} features={} O={}}},\
            lower-affine,\
            convert-arith-to-llvm,\
            convert-index-to-llvm{{index-bitwidth=64}},\
            canonicalize,\
            cse,\
            symbol-dce,\
            reconcile-unrealized-casts,\
            gpu.module(\
                convert-gpu-to-nvvm{{has-redux=1 use-bare-ptr-memref-call-conv=1}},\
                canonicalize,\
                cse),\
            gpu-to-llvm,\
            gpu-module-to-binary{{l={}}},\
            convert-math-to-llvm,\
            reconcile-unrealized-casts, canonicalize,cse)'",
            self.cubin_chip,
            self.cubin_features,
            self.opt_level,
            dev_lib.display()
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

    pub fn objtoptx(&self, ll_path: &Path, ptx_path: &Path) {
        fn extract_string_literal(line: &str) -> String {
            line[line.find("\"").unwrap() + 1..line.len() - 10].to_string()
        }
        fn unescape_llvm_string(s: &str) -> Vec<u8> {
            let mut bytes = Vec::new();
            let mut chars = s.chars().peekable();

            while let Some(ch) = chars.next() {
                if ch == '\\' {
                    // We expect two hex digits after '\'
                    let hi = chars.next().expect("Expected hex digit after \\");
                    if hi == '\\' {
                        // If we encounter another backslash, treat it as a literal backslash
                        bytes.push(b'\\');
                        continue;
                    }
                    let lo = chars.next().expect("Expected second hex digit after \\");
                    let hex_str = format!("{}{}", hi, lo);
                    let byte = u8::from_str_radix(&hex_str, 16)
                        .unwrap_or_else(|_| panic!("Invalid hex digits {}", hex_str));
                    bytes.push(byte);
                } else {
                    // Normal ASCII char
                    bytes.push(ch as u8);
                }
            }
            bytes
        }
        let content = std::fs::read_to_string(ll_path).expect("Failed to read .ll file");

        let ptx_data = content
            .lines()
            .find(|line| line.contains("@gpu_bin_cst"))
            .map(extract_string_literal)
            .unwrap();
        let bytes = unescape_llvm_string(&ptx_data);

        assert!(ptx_data.contains(".version"));
        assert!(ptx_data.contains(".target"));
        std::fs::write(ptx_path, bytes).expect("Failed to write PTX to file");
        let output = Command::new("cuobjdump")
            .arg("-ptx")
            .arg(ptx_path)
            .output()
            .map_err(|e| format!("Failed to run cuobjdump: {}", e))
            .unwrap();
        assert!(
            output.status.success(),
            "cuobjdump failed with status: {:?}\nError: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let ptx_str = String::from_utf8_lossy(&output.stdout).to_string();
        std::fs::write(ptx_path, &ptx_str).expect("Failed to write PTX to file");
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
        let ptx_out = inpath.with_extension("ptx");
        self.objtoptx(&llvmir_out, &ptx_out);
        Ok(())
    }
}
