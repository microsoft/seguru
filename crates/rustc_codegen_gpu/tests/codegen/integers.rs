// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
#[allow(unused_variables)]
pub unsafe fn integers(a: usize, b: i8, c: u8, d: u16, e: u32, out: &mut [usize]) {
    out[0] = a + b as usize;
    out[1] = a + 1;
    out[2] = (e + c as u32) as _;
    out[3] = a + d as usize;
    out[4] = a + d as usize;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry integers
// PTX_CHECK: st.global.u32 [%rd3]
// PTX_CHECK: st.global.u32 [%rd3+4]
// PTX_CHECK: st.global.u32 [%rd3+8]
// PTX_CHECK: st.global.u32 [%rd3+12]
// PTX_CHECK: st.global.u32 [%rd3+16]