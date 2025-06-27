fn main() {
    // Given MLIR filename, compile MLIR to object file
    let mlir_filename = std::env::args().nth(1).expect("No MLIR filename provided");
    let mlir_path = std::path::Path::new(&mlir_filename);
    let obj_path = mlir_path.with_extension("o");
    mlir_compile::CompileConfig::default()
        .mlir_compile(mlir_path, &obj_path)
        .expect("Failed to compile MLIR to object file");
}
