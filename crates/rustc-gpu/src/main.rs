use std::env;
use std::process::{Command, exit};

use log::{LevelFilter, debug};
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;

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
    let logfile = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
        .build("gpu_rustc.log")
        .unwrap();

    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .build(Root::builder().appender("logfile").build(LevelFilter::Debug))
        .unwrap();

    log4rs::init_config(config).unwrap();
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
