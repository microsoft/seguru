use std::env;
use std::process::{Command, exit};

use tracing::debug;

fn get_value(args: &[String], param: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(item) = iter.next() {
        if item == param {
            return iter.next().map(|s| s.to_string());
        }
    }
    None
}

fn main() -> std::io::Result<()> {
    // let log_file = std::fs::File::options().append(true).open("gpu_rustc.log")?;
    // tracing_subscriber::fmt().with_writer(std::sync::Mutex::new(log_file)).init();
    // crate1,crate2,..
    let gpu_target_str = env::var("GPU_TARGETS").unwrap_or_default();

    let gpu_targets: Vec<String> = gpu_target_str.split(',').map(|c| c.replace("-", "_")).collect();

    let mut args: Vec<String> = env::args().skip(1).collect();

    let rust_flags = env::var("RUSTFLAGS").unwrap_or_default();
    let codegen_path = env::var("GPU_CODEGEN").unwrap_or_default();
    let crate_name = get_value(&args, "--crate-name");
    debug!("gpu_targets = {:#?}, crate  = {:#?}", gpu_targets, crate_name);
    if let Some(crate_name) = crate_name {
        if gpu_targets.contains(&crate_name) {
            args.push("-Zcrate-attr=feature(register_tool)".to_string());
            args.push("-Zcrate-attr=register_tool(gpu_codegen)".to_string());
            args.push(format!("-Zcodegen-backend={}", codegen_path));
        }
    }
    run_rustc(&args, &rust_flags)?;

    Ok(())
}

fn run_rustc(args: &[String], rust_flags: &str) -> std::io::Result<()> {
    let mut command = Command::new("rustc");
    command.args(args);
    command.env("RUSTFLAGS", rust_flags);
    command.env("RUSTC_BOOTSTRAP", "1");
    debug!("cmd: {:?}", command);
    let status = command.status()?;
    if !status.success() {
        exit(status.code().unwrap_or(1));
    }

    Ok(())
}
