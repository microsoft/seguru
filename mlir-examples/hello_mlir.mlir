module attributes {gpu.container_module} {
  gpu.module @gpu attributes {visibility = "public"} {
    gpu.func @kernel_arith_wrapper(%arg0: memref<1xi8>, %arg1: i64, %arg2: i64, %arg3: memref<1xi8>, %arg4: i64, %arg5: i64) kernel attributes {sym_visibility = "private"} {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %1 = memref.get_global @memory_alloc_6 : memref<11xi8>
      %c11_2 = arith.constant 11 : index
      %thread_id_x = gpu.thread_id  x
      %2 = arith.index_cast %arg2 : i64 to index
      %3 = arith.muli %thread_id_x, %2 : index
      %subview = memref.subview %arg0[%3] [1] [1] : memref<1xi8> to memref<1xi8, strided<[1], offset: ?>>
      %4 = arith.index_cast %arg5 : i64 to index
      %5 = arith.muli %thread_id_x, %4 : index
      %subview_3 = memref.subview %arg3[%5] [1] [1] : memref<1xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_i64 = arith.constant 0 : i64
      %6 = arith.cmpi ult, %c0_i64, %arg2 : i64
      %c0_i64_4 = arith.constant 0 : i64
      %7 = arith.subi %c0_i64_4, %arg2 : i64
      %c63_i64 = arith.constant 63 : i64
      %8 = arith.shrsi %7, %c63_i64 : i64
      %intptr = memref.extract_aligned_pointer_as_index %alloca : memref<8xi8> -> index
      %9 = arith.index_cast %intptr : index to i64
      %10 = arith.andi %9, %8 : i64
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
      %12 = llvm.load volatile %11 {alignment = 8 : i64} : !llvm.ptr -> i8
      %subview_5 = memref.subview %subview[0] [1] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<1xi8, strided<[1], offset: ?>>
      %c0_6 = arith.constant 0 : index
      %13 = memref.load %subview_5[%c0_6] : memref<1xi8, strided<[1], offset: ?>>
      %c0_i64_7 = arith.constant 0 : i64
      %14 = arith.cmpi ult, %c0_i64_7, %arg5 : i64
      %c0_i64_8 = arith.constant 0 : i64
      %15 = arith.subi %c0_i64_8, %arg5 : i64
      %c63_i64_9 = arith.constant 63 : i64
      %16 = arith.shrsi %15, %c63_i64_9 : i64
      %intptr_10 = memref.extract_aligned_pointer_as_index %alloca : memref<8xi8> -> index
      %17 = arith.index_cast %intptr_10 : index to i64
      %18 = arith.andi %17, %16 : i64
      %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
      %20 = llvm.load volatile %19 {alignment = 8 : i64} : !llvm.ptr -> i8
      %subview_11 = memref.subview %subview_3[0] [1] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<1xi8, strided<[1], offset: ?>>
      %c0_12 = arith.constant 0 : index
      memref.store %13, %subview_11[%c0_12] : memref<1xi8, strided<[1], offset: ?>>
      gpu.return
    }
    memref.global constant @memory_alloc_6 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %0 = memref.get_global @memory_alloc_6 : memref<11xi8>
  }
  module {
  }
  %c11 = arith.constant 11 : index
  func.func private @_ZN3gpu20add_mlir_string_attr17hb6fab0ade4590ffcE(memref<1xi8>, i64) -> i64 attributes {gpu_codegen_builtin = "add_mlir_string_attr"}
  %f = func.constant {gpu_codegen_builtin = "add_mlir_string_attr"} @_ZN3gpu20add_mlir_string_attr17hb6fab0ade4590ffcE : (memref<1xi8>, i64) -> i64
  func.func private @_ZN3gpu9thread_id17h476e55ecb16338b5E() -> i64 attributes {gpu_codegen_builtin = "gpu.thread_id"}
  %f_0 = func.constant {gpu_codegen_builtin = "gpu.thread_id"} @_ZN3gpu9thread_id17h476e55ecb16338b5E : () -> i64
  %c0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
}
