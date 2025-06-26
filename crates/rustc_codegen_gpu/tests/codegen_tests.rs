// tests/codegen_tests.rs

use std::path::{Path, PathBuf};

use compiletest_rs as compiletest;

// suffix is .so or .rmeta
fn find_proc_macro_dylib(target_dir: &Path, crate_name: &str, suffix: &str) -> PathBuf {
    let search_dir = target_dir.join("deps");

    let entries = std::fs::read_dir(&search_dir).expect("Could not read deps dir");

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.starts_with(&format!("lib{}-", crate_name.replace("-", "_")))
            && file_name.ends_with(suffix)
        {
            return entry.path();
        }
    }

    panic!("Proc macro dylib for '{}' not found in {:?}", crate_name, search_dir);
}

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
    let gpu_macros_path = find_proc_macro_dylib(&target_dir, "gpu_macros", ".so");
    let gpu_path = find_proc_macro_dylib(&target_dir, "gpu", ".rlib");

    let config = compiletest_rs::Config {
        mode: mode.parse().unwrap(),
        src_base: src.clone(),
        build_base: target_dir.join(src.as_os_str()),
        llvm_filecheck: Some(target_dir.join("filecheck")),
        target_rustcflags: Some(format!(
            "-C opt-level=3 -Zcodegen-backend={} --crate-type=lib --extern gpu_macros={} --extern gpu={}",
            backend_path.display(),
            gpu_macros_path.display(),
            gpu_path.display()
        )),
        ..Default::default()
    };

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
