#![feature(rustc_private)]
#![feature(file_lock)]

extern crate rustc_ast;
extern crate rustc_codegen_llvm;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_lint_defs;
extern crate rustc_llvm;
extern crate rustc_middle;
extern crate rustc_parse;
extern crate rustc_resolve;
extern crate rustc_session;
extern crate rustc_span;

use std::process::Command;

mod codegen;
mod driver;

fn main() {
    let mut args: Vec<String> = std::env::args().collect();

    // Enable rust compiler tracing via RUSTC_LOG
    // See https://doc.rust-lang.org/stable/nightly-rustc/rustc_log/index.html
    let early_dcx =
        rustc_session::EarlyDiagCtxt::new(rustc_session::config::ErrorOutputType::default());
    rustc_driver::init_rustc_env_logger(&early_dcx);

    // Install a Ctrl+C handler
    rustc_driver::install_ctrlc_handler();

    let mut callbacks = crate::driver::GpuOrCpuRustCallback::default();

    if args.len() <= 1 {
    } else if let Ok(stage) = args[1].clone().try_into() {
        // This is a relaunch to compile Host code.
        callbacks.stage = stage;
        args.remove(1);
    }

    let mut exit_code =
        rustc_driver::catch_with_exit_code(|| rustc_driver::run_compiler(&args, &mut callbacks));
    if exit_code != 0 {
        early_dcx.early_fatal(format!("Failed at stage {:?}", callbacks.stage));
    }
    if let Some(next_stage) = callbacks.next_stage {
        args.insert(1, next_stage.into());
        exit_code = Command::new(&args[0]) // re-run self
            .args(&args[1..])
            .spawn()
            .expect("Failed to spawn child to compile gpu code for host execution")
            .wait()
            .expect("failed to compile host code")
            .code()
            .expect("");
    }
    std::process::exit(exit_code);
}
