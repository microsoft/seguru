// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;

use gpu::GPUDeviceFloatIntrinsics;

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_call_devicelib_func(out: &mut [f32], in1: f32, in2: f32) {
    let out = gpu::chunk_mut(out, 1, gpu::GpuChunkIdx::new());
    out[0] = in1.exp() + in2.sin() + in1.cosh() + in2.tanh() + in1.rsqrt() + in2.fma(in1, in2);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry intrinsics
