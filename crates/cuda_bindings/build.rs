extern crate bindgen;
extern crate cc;

use std::path::PathBuf;
use std::process::Command;

fn detect_cuda_version() -> (u32, u32) {
    let output = Command::new("nvcc").arg("--version").output().expect("Failed to run nvcc");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().find(|l| l.contains("release")).unwrap();
    let version = version_line.split("release").nth(1).unwrap().split(',').next().unwrap().trim();
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() != 2 {
        panic!("Invalid CUDA_VERSION format");
    }
    let major = parts[0].parse::<u32>().expect("Invalid major version");
    let minor = parts[1].parse::<u32>().expect("Invalid minor version");
    (major, minor)
}

fn main() {
    // Generate bindings
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MAX_THREAD_PER_BLOCK");
    println!("cargo:rerun-if-env-changed=MAX_BLOCK_DIM_X");
    println!("cargo:rerun-if-env-changed=MAX_BLOCK_DIM_Y");
    println!("cargo:rerun-if-env-changed=MAX_BLOCK_DIM_Z");
    println!("cargo:rerun-if-env-changed=MAX_GRID_DIM_X");
    println!("cargo:rerun-if-env-changed=MAX_GRID_DIM_Y");
    println!("cargo:rerun-if-env-changed=MAX_GRID_DIM_Z");
    println!("cargo:rerun-if-env-changed=MAX_SHARED_MEM_PER_BLOCK");
    println!("cargo:rerun-if-changed=libcuda_bindings/lib.h");
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("libcuda_bindings/lib.h")
        .use_core()
        .clang_arg("-I/usr/local/cuda/include")
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
        .generate_comments(false)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the src/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-lib=dylib=cuda");
    let (major, minor) = detect_cuda_version();
    println!("cargo:warning=CUDA version detected: {}.{}", major, minor);
    // Pass CUDA version as env variable to Rust code
    println!("cargo::rustc-check-cfg=cfg(cuda_has_ctx_create_v4)");
    if (major == 12 && minor >= 6) || major >= 13 {
        println!("cargo:rustc-cfg=cuda_has_ctx_create_v4");
    }
}
