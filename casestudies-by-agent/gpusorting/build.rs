use std::env;
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_BENCH").is_err() {
        return;
    }

    let cuda_dir = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_ref = format!("{manifest_dir}/cuda-ref/GPUSortingCUDA");

    // Compile DeviceRadixSort.cu (the kernel implementations)
    let nvcc = format!("{cuda_dir}/bin/nvcc");
    let common_args = ["-O3", "--compiler-options", "-fPIC", "-I", &cuda_ref, "-I", &format!("{cuda_dir}/include")];

    for (src, obj_name) in [
        (format!("{cuda_ref}/Sort/DeviceRadixSort.cu"), "DeviceRadixSort.o"),
        (format!("{manifest_dir}/cuda/sort_bench.cu"), "sort_bench.o"),
    ] {
        let obj_path = format!("{out_dir}/{obj_name}");
        let status = Command::new(&nvcc)
            .args(["-c", &src, "-o", &obj_path])
            .args(&common_args)
            .status()
            .unwrap_or_else(|e| panic!("failed to run nvcc on {src}: {e}"));
        assert!(status.success(), "nvcc failed on {src}");
    }

    // Create static library from both object files
    let lib_path = format!("{out_dir}/libsort_cuda.a");
    let status = Command::new("ar")
        .args([
            "rcs",
            &lib_path,
            &format!("{out_dir}/DeviceRadixSort.o"),
            &format!("{out_dir}/sort_bench.o"),
        ])
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-search=native={cuda_dir}/lib64");
    println!("cargo:rustc-link-lib=static=sort_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=cuda/sort_bench.cu");
}
