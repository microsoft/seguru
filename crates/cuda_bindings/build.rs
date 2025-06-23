extern crate bindgen;
extern crate cc;

use std::path::PathBuf;

fn main() {
    // Generate bindings
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=libcuda_bindings/lib.h");
    println!("cargo:rerun-if-changed=libcuda_bindings/lib.cu");
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("libcuda_bindings/lib.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the src/bindings.rs file.
    let out_path = PathBuf::from("src");
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");

    // Compile library
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .file("libcuda_bindings/lib.cu")
        .flag("-Wno-deprecated-gpu-targets")
        .shared_flag(true)
        .compile("cuda_bindings");

    println!("cargo:rustc-link-lib=cuda_bindings");
}
