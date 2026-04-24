fn main() {
    if std::env::var("CARGO_FEATURE_BENCH").is_ok() {
        let cuda_file = "cuda/bench_kernels.cu";
        println!("cargo:rerun-if-changed={cuda_file}");

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let obj_path = format!("{out_dir}/bench_kernels.o");
        let lib_path = format!("{out_dir}/libbench_kernels.a");

        let output = std::process::Command::new("nvcc")
            .args([
                "-c",
                cuda_file,
                "-o",
                &obj_path,
                "--compiler-options",
                "-fPIC",
                "-O3",
                "-arch=native",
            ])
            .output()
            .expect("Failed to run nvcc - is CUDA installed?");

        if !output.status.success() {
            panic!(
                "nvcc compilation failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let ar_output = std::process::Command::new("ar")
            .args(["rcs", &lib_path, &obj_path])
            .output()
            .expect("Failed to run ar");

        if !ar_output.status.success() {
            panic!("ar failed:\n{}", String::from_utf8_lossy(&ar_output.stderr));
        }

        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static=bench_kernels");

        // Link CUDA runtime
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
