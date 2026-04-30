use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda_dir = "/usr/local/cuda";
    let cuda_src = "cuda/train_gpt2_fp32.cu";
    let out_dir = env::var("OUT_DIR").unwrap();
    let cur_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let llmc_dir = format!("{}/../../", cur_dir);
    // cd ../
    // make libtrain_gpt2fp32.a
    let use_llvm = if cfg!(feature = "llvm") { "1" } else { "0" };
    // remove libtrain_gpt2fp32.a if exists
    let _ = Command::new("rm")
        .current_dir(&llmc_dir)
        .arg("-f")
        .arg("libtrain_gpt2fp32.a")
        .status()
        .expect("Failed to execute rm command");
    let _ = Command::new("rm")
        .current_dir(&llmc_dir)
        .arg("-f")
        .arg("libtrain_gpt2fp32.o")
        .status()
        .expect("Failed to execute rm command");
    let status = Command::new("make")
        .current_dir(&llmc_dir)
        .env("USE_LLVM", use_llvm)
        .arg("libtrain_gpt2fp32.a")
        .status()
        .expect("Failed to execute make command");
    assert!(status.success(), "Make command failed");

    let header = format!("{}/train_gpt2_fp32.h", llmc_dir);
    let src = format!("{}/train_gpt2_fp32.cu", llmc_dir);

    // 2️⃣ Tell Cargo to link the static library
    println!("cargo:rustc-link-search=native={}", llmc_dir);
    println!("cargo:rustc-link-search=native={}/lib64", cuda_dir);
    println!("cargo:rustc-link-search=native={}/lib64/stubs", cuda_dir);
    println!("cargo:rustc-link-lib=static=train_gpt2fp32");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={}", cuda_src);
    println!("cargo:rerun-if-changed={}", header);
    println!("cargo:rerun-if-changed={}", src);
    println!("cargo:rerun-if-changed=build.rs");

    // Generate Rust bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header(header.as_str())
        .generate()
        .expect("Unable to generate bindings");

    let bindings_out = PathBuf::from(out_dir).join("train_gpt2_fp32_bindings.rs");
    bindings.write_to_file(&bindings_out).expect("Couldn't write bindings!");
}
