use std::env;
use std::process::Command;

fn main() {
    // Only compile CUDA when the bench feature is active
    if env::var("CARGO_FEATURE_BENCH").is_err() {
        return;
    }

    let cuda_dir = "/usr/local/cuda";
    let out_dir = env::var("OUT_DIR").unwrap();
    let cur_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let cu_src = format!("{}/cuda/kernels.cu", cur_dir);
    let obj_path = format!("{}/kernels.o", out_dir);
    let lib_path = format!("{}/libkernelbench_cuda.a", out_dir);

    // Compile .cu → .o
    let status = Command::new(format!("{}/bin/nvcc", cuda_dir))
        .args([
            "-c",
            &cu_src,
            "-o",
            &obj_path,
            "-O2",
            "--compiler-options",
            "-fPIC",
            "-I",
            &format!("{}/cuda/", cur_dir),
        ])
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "nvcc compilation failed");

    // Create static library
    let status = Command::new("ar")
        .args(["rcs", &lib_path, &obj_path])
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    // Link directives
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native={}/lib64", cuda_dir);
    println!("cargo:rustc-link-lib=static=kernelbench_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/kernels.h");
}
