// compile-flags: --emit=llvm-ir -C llvm-args="--fp-contract=fast" -C llvm-args="--denormal-fp-math=preserve-sign" -C llvm-args="--denormal-fp-math-f32=preserve-sign"
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;
use gpu::Float4;

#[gpu::device]
#[inline(always)]
#[no_mangle]
pub fn compute_on_float4(a: Float4, b: Float4) -> Float4 {
    let f: f32 = 1.234;
    Float4::new([
        a.data[0] + b.data[0] + f,
        a.data[1] + b.data[1],
        a.data[2] + b.data[2],
        a.data[3] / b.data[3],
    ])
}

#[gpu::kernel]
#[no_mangle]
pub fn test_float4(out: &mut [Float4], in2: &Float4) {
    let in1 = Float4::new([1.234, 5.678, 9.1011, 12.1314]);
    let mut out = gpu::chunk_mut(out, gpu::MapLinear::new(1));
    out[0] = compute_on_float4(in1, *in2);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry float
// PTX_CHECK: add.ftz.f32
// PTX_CHECK: div.approx.ftz.f32
// PTX_CHECK: 0f3F9DF3B6
// PTX_CHECK: ld.global.v4.f32
// PTX_CHECK: st.global.v4.f32
