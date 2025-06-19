// tests/codegen_tests.rs

use std::path::PathBuf;

use compiletest_rs as compiletest;

fn run_codegen_tests(src: PathBuf, mode: &str) {
    #[cfg(debug_assertions)]
    let profile = "debug";
    #[cfg(not(debug_assertions))]
    let profile = "release";
    let target_dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target").join(profile)
    });
    let mut config = compiletest::Config::default();
    config.mode = mode.parse().unwrap();
    config.src_base = src.clone();
    config.build_base = target_dir.join(src.as_os_str());
    config.llvm_filecheck = Some(target_dir.join("filecheck"));

    // 👇 Point to your shared backend library
    let backend_path = target_dir.join("librustc_codegen_gpu.so");
    config.target_rustcflags = Some(format!(
        "-C opt-level=3 -Zcodegen-backend={} --crate-type=lib",
        backend_path.display()
    ));
    println!("config.target_rustcflags: {}", config.target_rustcflags.as_ref().unwrap());
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
