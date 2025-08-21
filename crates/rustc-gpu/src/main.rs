use std::env;
use std::path::PathBuf;
use std::process::{Command, exit};

use tracing::debug;

fn get_values(args: &[String], param: &str) -> Vec<String> {
    let mut iter = args.iter();
    let mut ret = vec![];
    while let Some(item) = iter.next() {
        if item == param {
            ret.push(iter.next().unwrap().to_string());
        }
    }
    ret
}

fn get_value(args: &[String], param: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(item) = iter.next() {
        if item == param {
            return iter.next().map(|s| s.to_string());
        }
    }
    None
}

fn get_value_c(args: &[String], param: &str) -> Option<String> {
    let iter = args.iter();
    for item in iter {
        if item.starts_with(param) {
            return Some(item.strip_prefix(param).unwrap().to_string());
        }
    }
    None
}

fn set_value_c(args: &mut Vec<String>, param: &str, value: &str) {
    if let Some(pos) = args.iter().position(|x| x.starts_with(param)) {
        args[pos] = format!("{}{}", param, value);
    } else {
        args.push(format!("{}{}", param, value));
    }
}

fn main() -> std::io::Result<()> {
    let log_file = std::fs::File::options().append(true).create(true).open("/tmp/rustc-gpu.log")?;
    tracing_subscriber::fmt().with_writer(std::sync::Mutex::new(log_file)).init();
    // crate1,crate2,..
    let gpu_target_str = env::var("GPU_TARGETS").unwrap_or_default();

    let gpu_targets: Vec<String> = gpu_target_str.split(',').map(|c| c.replace("-", "_")).collect();

    let mut args: Vec<String> = env::args().skip(1).collect();

    let rust_flags = env::var("RUSTFLAGS").unwrap_or_default();
    let codegen_path = env::var("GPU_CODEGEN").unwrap_or_default();
    let crate_names = get_value(&args, "--crate-name");
    debug!("gpu_targets = {:#?}, crate  = {:#?}", gpu_targets, crate_names);
    if let Some(crate_name) = crate_names {
        let mut host_args = args.clone();
        if gpu_targets.contains(&crate_name) {
            let extra_args = vec![
                "-Zcrate-attr=feature(register_tool)".into(),
                "-Zcrate-attr=register_tool(gpu_codegen)".into(),
            ];
            args.push("--cfg".to_string());
            args.push("gpu_code".to_string());
            for arg in extra_args {
                if !args.contains(&arg) {
                    args.push(arg);
                }
            }

            // Compile the code with GPU backend.
            // Only generating obj files.
            set_value_c(&mut args, "--emit=", "obj");
            let out_dir = get_value(&args, "--out-dir").unwrap();
            let out_dir = std::path::Path::new(&out_dir);
            let extra_filename = get_value_c(&args, "extra-filename=").unwrap();
            let extra_filename = format!("{}-gpu", extra_filename);
            let bc_file = out_dir.join(format!("lib{}{}.gpu.bc", crate_name, extra_filename));
            let metadata = get_value_c(&args, "metadata=").unwrap_or_default();
            let metadata = format!("{}-gpu", metadata);
            set_value_c(&mut args, "extra-filename=", &extra_filename);
            set_value_c(&mut args, "metadata=", &metadata);

            // Compile the GPU code with GPU backend.
            args.push(format!("-Zcodegen-backend={}", codegen_path));
            unsafe {
                env::set_var("GPU_HOST_OBJ", format!("{}{}.o", crate_name, extra_filename));
            }
            run_rustc(&args, &rust_flags, "GPU")?;
            let crate_type = &get_value(&args, "--crate-type").unwrap();
            let externs = get_values(&args, "--extern");
            let extern_gpu_bc_files = externs.iter().map(|s| {
                let p = std::path::Path::new(s.split("=").nth(1).unwrap());
                p.parent()
                    .unwrap()
                    .join(format!("{}-gpu", p.file_stem().unwrap().to_str().unwrap()))
                    .with_extension("gpu.bc")
            });

            let mut bc_files: Vec<PathBuf> = extern_gpu_bc_files.filter(|s| s.exists()).collect();
            if bc_file.exists() {
                bc_files.push(bc_file);
            }
            if crate_type == "bin" {
                let libname = format!("{}{}", crate_name, extra_filename);
                let gpu_obj_file = out_dir.join(format!("lib{}.a", libname));
                mlir_compile::CompileConfig::new()
                    .gpu_link_and_create_static_lib(&bc_files, &gpu_obj_file)?;
                host_args.extend([
                    "-L".into(),
                    out_dir.to_str().unwrap().into(),
                    "-l".into(),
                    libname,
                ]);
            }
            run_rustc(&host_args, &rust_flags, "CPU")?;
        } else {
            run_rustc(&args, &rust_flags, "CPU")?;
        }
    } else {
        run_rustc(&args, &rust_flags, "CPU")?;
    }

    Ok(())
}

fn run_rustc(args: &[String], rust_flags: &str, target: &str) -> std::io::Result<()> {
    let mut command = Command::new("rustc");
    command.args(args);
    command.env("RUSTFLAGS", rust_flags);
    command.env("RUSTC_BOOTSTRAP", "1");
    command.env("__CODEGEN_TARGET__", target);
    debug!("cmd: {:?}", command);
    let status = command.status()?;
    if !status.success() {
        exit(status.code().unwrap_or(1));
    }

    Ok(())
}
