use std::process::Command;

fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-lib=cuda");

    // An ultra-terrible way to do this. We can't let Cargo to create that static
    // library for us because it will pack shits into it. We only need that .o
    // file so we'll have to pack the .a on our own
    let target_path = format!("{}/../../../../release/", std::env::var("OUT_DIR").unwrap());
    let output_lib_path = format!("{}/libmanual_test_gpu_arith.a", target_path);
    let obj_lib_path = format!("{}/deps/manual_test_gpu_arith-*.0.rcgu.o", target_path);
    let ar_cmd = format!(
        "/home/linuxbrew/.linuxbrew/opt/binutils/bin/ar rcs {} {}",
        output_lib_path, obj_lib_path
    );

    Command::new("/bin/bash").args(["-c", &ar_cmd]).output().expect("failed to execute process");

    println!("cargo:rustc-link-search=native={}", target_path);
    println!("cargo:rustc-link-lib=manual_test_gpu_arith");
}
