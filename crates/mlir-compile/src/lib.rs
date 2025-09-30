use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::info;

#[derive(Debug)]
pub struct CompileConfig {
    pub opt_level: u8,
    pub cubin_chip: String,
    pub cubin_features: String,
    pub gpu_sym: String,
    pub use_fast: bool,
    pub use_ftz: bool,
    pub dep_device_bc_files: Vec<PathBuf>,
    pub llc_ptx_extra: Vec<String>,
}

impl Default for CompileConfig {
    fn default() -> Self {
        CompileConfig {
            opt_level: 3,
            cubin_chip: "sm_80".to_string(),
            cubin_features: "+ptx80".to_string(),
            gpu_sym: "gpu_bin_cst".to_string(),
            use_fast: true,
            use_ftz: true,
            dep_device_bc_files: vec![find_libdevice().unwrap()],
            llc_ptx_extra: vec![],
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
    pub fn new() -> CompileConfig {
        let mut cconfig = CompileConfig::default();
        std::env::var("USE_FAST").map(|v| cconfig.use_fast = v == "true").unwrap_or_default();
        std::env::var("USE_FTZ").map(|v| cconfig.use_ftz = v == "true").unwrap_or_default();
        std::env::var("NVPTX_ARCH").map(|v| cconfig.cubin_chip = v).unwrap_or_default();
        std::env::var("NVPTX_FEATURES").map(|v| cconfig.cubin_features = v).unwrap_or_default();
        std::env::var("PTXAS_OPT_LEVEL")
            .map(|v| cconfig.opt_level = v.parse().unwrap_or_default())
            .unwrap_or_default();
        // TODO(dep): add more gpu device files to dep_device_bc_files
        cconfig
    }

    fn opt_flag(&self) -> String {
        format!("-O{}", self.opt_level)
    }

    fn gpu_to_bin_args(&self) -> String {
        let dev_libs = self
            .dep_device_bc_files
            .iter()
            .map(|p| p.to_str().unwrap())
            .collect::<Vec<_>>()
            .join(",");
        format!("format=llvm l={}", dev_libs)
        //format!("format=fatbin l={}", dev_libs)
    }

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
            mem2reg,\
            nvvm-attach-target{{triple=nvptx64-nvidia-gpulibs {} {} chip={} features={} O={}}},\
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
            gpu-module-to-binary{{{}}},\
            convert-math-to-llvm,\
            reconcile-unrealized-casts, canonicalize,cse)'",
            if self.use_fast { "fast" } else { "" },
            if self.use_ftz { "ftz" } else { "" },
            self.cubin_chip,
            self.cubin_features,
            self.opt_level,
            self.gpu_to_bin_args(),
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

    pub fn llvm_link(&self, inpath: &[PathBuf], outpath: &Path) -> std::io::Result<()> {
        info!("[llvm-link] {:?} outputs {}", inpath, outpath.display());
        let mut args = inpath.iter().map(|p| p.to_str().unwrap()).collect::<Vec<_>>();
        args.extend(["-o", outpath.to_str().unwrap()]);
        command("llvm-link", args)
    }

    fn llc_ptx(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        info!("[llc] {:?} outputs {:?}", inpath, outpath);
        let mut args = vec![
            "-march=nvptx64".into(),
            format!("-mcpu={}", self.cubin_chip),
            format!("-mattr={}", self.cubin_features),
            self.opt_flag(),
            inpath.to_str().unwrap().into(),
            "-o".into(),
            outpath.to_str().unwrap().into(),
        ];
        if self.use_fast {
            args.push("--fp-contract=fast".into());
            args.push("--nvptx-prec-divf32=0".into());
            args.push("--nvptx-approx-log2f32".into());
            args.push("--nvptx-prec-sqrtf32=0".into());
            args.push("--nvptx-rsqrt-approx-opt".into());
        }
        if self.use_ftz {
            args.push("--denormal-fp-math-f32=preserve-sign".into());
            args.push("--denormal-fp-math=preserve-sign".into());
        }
        command("llc", args)
    }

    fn ptxas(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        info!("[ptxas] outputs {}", outpath.display());
        let args = [
            &self.opt_flag(),
            "--gpu-name",
            &self.cubin_chip,
            "-o",
            outpath.to_str().unwrap(),
            inpath.to_str().unwrap(),
        ];
        command("ptxas", args)
    }

    fn fatbinary(&self, cubin_path: &Path, ptx_path: &Path, outpath: &Path) -> std::io::Result<()> {
        info!("[fatbinary] outputs {}", outpath.display());
        let sm = self.cubin_chip.strip_prefix("sm_").unwrap();
        let args = [
            "-64",
            &format!("--image3=kind=elf,sm={},file={}", sm, cubin_path.to_str().unwrap()),
            &format!("--image3=kind=ptx,sm={},file={}", sm, ptx_path.to_str().unwrap()),
            &format!("--create={}", outpath.to_str().unwrap()),
        ];
        command("fatbinary", args)
    }

    fn create_obj(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        let old_sym =
            inpath.display().to_string().replace("/", "_").replace("-", "_").replace(".", "_");
        info!("[objcopy] {} outputs {}", old_sym, outpath.display());
        let args = [
            "--input",
            "binary",
            "--output",
            "elf64-x86-64",
            "--binary-architecture",
            "i386:x86-64",
            "--rename-section",
            ".data=.rodata",
            "--redefine-sym",
            &format!("_binary_{}_start={}", old_sym, self.gpu_sym),
            inpath.to_str().unwrap(),
            outpath.to_str().unwrap(),
        ];
        command("objcopy", args)
    }

    fn create_static_lib(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        info!("[llvm-ar] rcS {} {}", outpath.display(), inpath.display());
        command("llvm-ar", ["rcS", outpath.to_str().unwrap(), inpath.to_str().unwrap()])
    }

    pub fn ld(&self, inpath: &[PathBuf], outpath: &Path) -> std::io::Result<()> {
        let mut args = inpath.iter().map(|p| p.to_str().unwrap()).collect::<Vec<_>>();
        args.extend(&["-r", "-o", outpath.to_str().unwrap()].to_vec());
        command("ld", args)
    }

    pub fn expose_gpu_obj(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        info!("[objcopy] outputs {}", outpath.display());
        let flag = format!("--globalize-symbol={}", self.gpu_sym);
        let args = [flag.as_str(), inpath.to_str().unwrap(), outpath.to_str().unwrap()];
        command("objcopy", args)
    }

    fn extract_gpu_bin(&self, ll_path: &Path, out_path: &Path) -> bool {
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

        let Some(bin_data) =
            content.lines().find(|line| line.contains("@gpu_bin_cst")).map(extract_string_literal)
        else {
            std::fs::write(out_path, vec![]).expect("Failed to write bin to file");
            return false;
        };
        let bytes = unescape_llvm_string(&bin_data);
        std::fs::write(out_path, bytes).expect("Failed to write bin to file");
        true
    }

    #[allow(dead_code)]
    fn cuobjdump(&self, cubin_path: &Path, ptx_path: &Path) {
        let output = Command::new("cuobjdump")
            .arg("-ptx")
            .arg(cubin_path)
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

    /// Create a static library from gpu bitcode files.
    /// The final cubin will be stored at self.gpu_sym.
    pub fn gpu_link_and_create_static_lib(
        &self,
        bc_files: &[PathBuf],
        out_path: &Path,
    ) -> std::io::Result<Option<PathBuf>> {
        let mut gpu_byte_code_files = vec![];
        let bitcode_out = out_path.with_extension("gpu.bc");
        let cubin_out = out_path.with_extension("cubin");
        let fatbin_out = out_path.with_extension("fatbin");
        let ptx_out = out_path.with_extension("ptx");
        let tmp_obj = out_path.with_extension("tmp.o");
        for bc_file in bc_files {
            gpu_byte_code_files.push(bc_file.clone());
        }
        self.llvm_link(&gpu_byte_code_files, &bitcode_out)?;
        self.llc_ptx(&bitcode_out, &ptx_out)?;
        self.ptxas(&ptx_out, &cubin_out)?;
        self.fatbinary(&cubin_out, &ptx_out, &fatbin_out)?;
        self.create_obj(&fatbin_out, &tmp_obj)?;
        self.create_static_lib(&tmp_obj, out_path)?;
        assert!(bitcode_out.exists());
        Ok(Some(bitcode_out))
    }

    pub fn mlir_compile(&self, inpath: &Path, outpath: &Path) -> std::io::Result<()> {
        let opt_out = inpath.with_extension("opt.mlir.mir");
        let llvmir_out = inpath.with_extension("ll");
        let bitcode_out = inpath.with_extension("bc");
        let gpu_obj_out = inpath.with_extension("o");
        let gpu_bitcode_out = inpath.with_extension("gpu.bc");
        let ptx_out = inpath.with_extension("ptx");
        let final_out = outpath.with_extension("o");

        self.mlir_opt(inpath, &opt_out)?;
        self.mlir_translate(&opt_out, &llvmir_out)?;
        self.llvm_as(&llvmir_out, &bitcode_out)?;
        self.llc(&bitcode_out, &gpu_obj_out)?;

        // extract the bc, since fmt is llvm
        let has_gpu_code = self.extract_gpu_bin(&llvmir_out, &gpu_bitcode_out);
        if has_gpu_code {
            self.llc_ptx(&gpu_bitcode_out, &ptx_out)?;
        }

        self.expose_gpu_obj(&gpu_obj_out, &final_out)
    }
}
