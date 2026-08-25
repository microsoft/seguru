use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/sort_ref.cu");
    if env::var("CARGO_FEATURE_BENCH").is_err() {
        return;
    }

    let cuda_dir = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let src = manifest.join("cuda/sort_ref.cu");
    let obj = out_dir.join("sort_ref.o");
    let lib = out_dir.join("libsort_ref.a");

    let status = Command::new(format!("{cuda_dir}/bin/nvcc"))
        .args(["-c", src.to_str().unwrap(), "-o", obj.to_str().unwrap()])
        .args(["-O3", "-lineinfo", "-arch=native", "-std=c++17", "--extended-lambda", "--compiler-options", "-fPIC"])
        .status()
        .expect("failed to invoke nvcc; set CUDA_PATH or disable the `bench` feature");
    assert!(status.success(), "nvcc failed to compile cuda/sort_ref.cu");

    let status = Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("failed to invoke ar");
    assert!(status.success(), "ar failed to archive the CUDA reference");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sort_ref");
    println!("cargo:rustc-link-search=native={cuda_dir}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
