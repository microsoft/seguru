extern crate bindgen;
extern crate cc;

use std::path::PathBuf;

fn main() {
    // Generate bindings
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=build.rs");
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
}
