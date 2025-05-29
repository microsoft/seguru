fn main() {
    println!(
        "cargo:rustc-link-search=/home/linuxbrew/.linuxbrew/lib\n\
    cargo:rustc-link-search=/home/linuxbrew/.linuxbrew/opt/llvm/lib"
    );
}
