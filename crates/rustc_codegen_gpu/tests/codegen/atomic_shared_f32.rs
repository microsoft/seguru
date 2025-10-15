// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu::kernel(dynamic_shared)]
#[no_mangle]
pub fn test_atomic_shared(f: f32, b: &mut f32) {
    let mut smem = smem_alloc.alloc::<f32>(1);
    let atomic_smem = gpu::sync::SharedAtomic::new(&mut smem);
    atomic_smem.index(0).atomic_addf(f);
    let b_atomic = gpu::sync::Atomic::new(b);
    gpu::sync_threads();
    b_atomic.atomic_addf(f);
    b_atomic.atomic_addf(*smem[0]);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry atomic_
// PTX_CHECK: atom.global.add.f32
// PTX_CHECK: atom.shared.add.f32

