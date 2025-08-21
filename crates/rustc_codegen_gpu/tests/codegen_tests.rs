// tests/codegen_tests.rs

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use compiletest_rs as compiletest;

static INIT: Once = Once::new();

// suffix is .so or .rmeta
fn find_lib(search_dir: &Path, lib_name: &str, suffix: &str) -> PathBuf {
    let entries = std::fs::read_dir(search_dir)
        .unwrap_or_else(|_| panic!("Could not read deps dir {}", search_dir.display()));

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.starts_with(lib_name) && file_name.ends_with(suffix) {
            return entry.path();
        }
    }

    panic!("Proc macro dylib for '{}' not found in {:?}", lib_name, search_dir);
}

//pub const TARGET:&str = "nvptx64-nvidia-cuda";
pub const TARGET: &str = "x86_64-unknown-linux-gnu";

fn run_codegen_tests(src: PathBuf, mode: &str) {
    #[cfg(debug_assertions)]
    let profile = "debug";
    #[cfg(not(debug_assertions))]
    let profile = "release";
    let target_dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target").join(profile)
    });

    // 👇 Point to your shared backend library
    let backend_path = target_dir.join("librustc_codegen_gpu.so");
    let gpu_macros_path = find_lib(&target_dir.join("deps"), "libgpu_macros-", ".so");
    let gpu_macros = format!("gpu_macros={}", &gpu_macros_path.to_str().unwrap());

    let gpu_src = target_dir.join("../../gpu/src/lib.rs");
    let gpu_target = target_dir.join("tests/gpu");
    let codegen = format!("-Zcodegen-backend={}", backend_path.to_str().unwrap());
    let rustc_flags = vec![
        "-C",
        "opt-level=3",
        "-C",
        "codegen-units=1",
        "--crate-type=lib",
        "--extern",
        gpu_macros.as_str(),
        "--cfg",
        "gpu_codegen",
        &codegen,
    ];
    let mut rustc_gpu_flags = rustc_flags.clone();
    rustc_gpu_flags.extend([
        "--target",
        TARGET,
        "--out-dir",
        gpu_target.to_str().unwrap(),
        "--cfg",
        "feature=\"codegen_tests\"",
        "--crate-name",
        "gpu",
        gpu_src.to_str().unwrap(),
    ]);
    println!("rustc_gpu_flags = rustc {:?}", rustc_gpu_flags.join(" "));
    INIT.call_once(|| {
        Command::new("rustc").args(&rustc_gpu_flags).status().expect("Failed to run rustc");
    });
    INIT.wait();
    let gpu_path = find_lib(&gpu_target, "libgpu", ".rlib");
    let rustc_test_flags = format!("{} --extern gpu={}", rustc_flags.join(" "), gpu_path.display());
    let config = compiletest_rs::Config {
        mode: mode.parse().unwrap(),
        src_base: src.clone(),
        build_base: target_dir.join(src.as_os_str()),
        llvm_filecheck: Some(target_dir.join("filecheck")),
        target_rustcflags: Some(rustc_test_flags),
        target: TARGET.into(),
        ..Default::default()
    };
    std::env::set_var("__CODEGEN_TARGET__", "GPU");
    compiletest::run_tests(&config);
}

#[test]
fn test_codegen_backend_output() {
    run_codegen_tests(PathBuf::from("tests/codegen"), "codegen");
}

#[test]
fn test_codegen_backend_output_fails() {
    run_codegen_tests(PathBuf::from("tests/codegen_fails"), "compile-fail");
}
