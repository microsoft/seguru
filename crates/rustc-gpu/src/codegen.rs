use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::OnceLock;

pub const CODEGEN_FILENAME: &str = "librustc_codegen_gpu.so";
pub const CODEGEN_DYLIB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/../../../librustc_codegen_gpu.so"));

pub(crate) fn get_codegen_dylib(out_dir: &Path) -> String {
    static CODEGEN_PATH: OnceLock<String> = OnceLock::new();

    CODEGEN_PATH
        .get_or_init(|| {
            let so_path = out_dir.join(CODEGEN_FILENAME);
            let mut file = OpenOptions::new()
                .write(true)
                .read(true)
                .create(true)
                .open(&so_path)
                .expect("Failed to open file");

            // lock file to ensure no concurrent read/write to the file
            // `cargo build` may run multiple instances in parallel
            file.lock().expect("Failed to lock file");
            let mut content: Vec<u8> = vec![];
            let size = file.read(&mut content).expect("failed to read CODEGEN_DYLIB");
            if size != CODEGEN_DYLIB.len() || content != CODEGEN_DYLIB {
                let size = file.write(CODEGEN_DYLIB).expect("failed to write CODEGEN_DYLIB");
                assert!(size == CODEGEN_DYLIB.len());
            }
            file.unlock().expect("Failed to unlock file");

            so_path.to_str().unwrap().into()
        })
        .clone()
}
