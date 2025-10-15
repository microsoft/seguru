// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu::kernel]
#[no_mangle]
pub fn test_atomic_shared_failed(b: &mut f32) {
    let mut smem = gpu::GpuShared::<f32>::zero();
    let atomic_smem = gpu::sync::SharedAtomic::new(&mut smem); //~ ERROR The write needs a `sync_threads` called before other read/write
    atomic_smem.atomic_addf(1.0);
    let b_atomic = gpu::sync::Atomic::new(b);
    b_atomic.atomic_addf(*smem); //~ NOTE need `sync_threads` before this read/write
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry atomic_
// PTX_CHECK: atom.global.cas.b32
// PTX_CHECK: atom.shared.cas.b32

