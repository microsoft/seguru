#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub fn kernel_print() {
    gpu::add_mlir_string_attr("\"run\"");
    gpu::printf();
    let _ = gpu::thread_id(gpu::DimType::X);
}
