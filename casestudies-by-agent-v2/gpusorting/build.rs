use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/sort_ref.cu");
    println!("cargo:rerun-if-changed=cuda/drs_variant.cu");
    println!("cargo:rerun-if-changed=cuda/upstream/DeviceRadixSort.cu");
    if env::var("CARGO_FEATURE_BENCH").is_err() {
        return;
    }

    let cuda_dir = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let lib = out_dir.join("libsort_ref.a");

    // The same-algorithm baseline is the *same* translation unit compiled twice
    // with different tuning macros: once as upstream tunes it (7680 keys per
    // tile, 15 per thread) and once at our Rust port's tuning (4096 / 8). The
    // namespace is renamed by macro so both variants can be linked together.
    let ours_tuning = [
        "-DPART_SIZE=4096",
        "-DVEC_PART_SIZE=1024",
        "-DBIN_PART_SIZE=4096",
        "-DBIN_SUB_PART_SIZE=256",
        "-DBIN_KEYS_PER_THREAD=8",
    ];
    let mut objs = Vec::new();
    let units: [(&str, &str, &[&str]); 3] = [
        ("cuda/sort_ref.cu", "sort_ref.o", &[]),
        (
            "cuda/drs_variant.cu",
            "drs_up.o",
            &["-DDeviceRadixSort=DrsUp", "-DDRS_DISPATCH=drs_dispatch_up"],
        ),
        (
            "cuda/drs_variant.cu",
            "drs_ours.o",
            &["-DDeviceRadixSort=DrsOurs", "-DDRS_DISPATCH=drs_dispatch_ours"],
        ),
    ];
    for (src_rel, obj_name, extra) in units {
        let src = manifest.join(src_rel);
        let obj = out_dir.join(obj_name);
        let mut cmd = Command::new(format!("{cuda_dir}/bin/nvcc"));
        cmd.args(["-c", src.to_str().unwrap(), "-o", obj.to_str().unwrap()])
            .args(["-O3", "-lineinfo", "-arch=native", "-std=c++17", "--extended-lambda", "--compiler-options", "-fPIC"])
            .args(extra);
        if obj_name == "drs_ours.o" {
            cmd.args(ours_tuning);
        }
        let status = cmd
            .status()
            .expect("failed to invoke nvcc; set CUDA_PATH or disable the `bench` feature");
        assert!(status.success(), "nvcc failed to compile {src_rel} -> {obj_name}");
        objs.push(obj);
    }

    let mut ar = Command::new("ar");
    ar.args(["rcs", lib.to_str().unwrap()]);
    for o in &objs {
        ar.arg(o);
    }
    let status = ar.status().expect("failed to invoke ar");
    assert!(status.success(), "ar failed to archive the CUDA reference");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sort_ref");
    println!("cargo:rustc-link-search=native={cuda_dir}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
