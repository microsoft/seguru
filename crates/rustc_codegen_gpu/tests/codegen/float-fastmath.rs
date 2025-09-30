// compile-flags: --emit=llvm-ir -C llvm-args="--fp-contract=fast" -C llvm-args="--denormal-fp-math=preserve-sign" -C llvm-args="--denormal-fp-math-f32=preserve-sign"
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;
use gpu::float4;

#[gpu_macros::device]
#[inline(always)]
#[no_mangle]
pub fn add_float4(a: &float4, b: &float4) -> float4 {
    let f: f32 = 1.234;
    float4 {
        x: a.x + b.x + f,
        y: a.y + b.y,
        z: a.z + b.z,
        w: a.w / b.w,
    }
}

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_float4(out: &mut [float4], in1: &float4, in2: &float4) {
    let mut out = gpu::chunk_mut(out, gpu::MapLinear::new(1));
    out[0] = add_float4(in1, in2);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry float
// PTX_CHECK: add.ftz.f32
// PTX_CHECK: div.approx.ftz.f32
// PTX_CHECK: 0f3F9DF3B6
