// compile-flags: -C llvm-args=--fp-contract=contract -C llvm-args=--denormal-fp-math=ieee -C llvm-args=--denormal-fp-math-f32=ieee --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;
use gpu::Float4;

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_float4(out: &mut [Float4], in2: &Float4) {
    let in1 = Float4::new([1.234, 5.678, 9.1011, 12.1314]);
    let mut out = gpu::chunk_mut(out, gpu::MapLinear::new(1));
    out[0] = in1 + *in2;
}
// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry float4
// PTX_CHECK: 0f3F9DF3B6
// PTX_CHECK: ld.global.v4.f32
// PTX_CHECK: st.global.v4.f32
// PTX_CHECK: add.rn.f32

