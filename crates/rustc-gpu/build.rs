fn main() {
    let rustc = std::env::var("RUSTC").unwrap();
    let output = std::process::Command::new(rustc).arg("--print=sysroot").output();
    let stdout = String::from_utf8(output.unwrap().stdout).unwrap();
    let sysroot = stdout.trim_end();
    // Ensure the sysroot "lib" directory is added to the binary rpath so
    // compiler-provided libraries (e.g. rustc_driver) can be found at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath={sysroot}/lib")
}
