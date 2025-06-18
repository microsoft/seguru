module attributes {gpu.container_module} {
  gpu.module @gpu attributes {visibility = "public"} {
    gpu.func @kernel_arith_wrapper(%arg0: memref<1xi8>, %arg1: i64, %arg2: index, %arg3: memref<1xi8>, %arg4: i64, %arg5: index) kernel attributes {sym_visibility = "private"} {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %1 = memref.get_global @memory_alloc_2 : memref<11xi8>
      %c11_4 = arith.constant 11 : index
      %thread_id_x = gpu.thread_id  x
      %2 = arith.muli %thread_id_x, %arg2 : index
      %3 = arith.addi %arg2, %2 : index
      %c1_i64 = arith.constant 1 : i64
      %c1 = arith.constant 1 : index
      %4 = arith.subi %3, %c1 : index
      %5 = arith.index_cast %arg1 : i64 to index
      %6 = arith.subi %4, %5 : index
      %c63_i64 = arith.constant 63 : i64
      %c63 = arith.constant 63 : index
      %7 = arith.shrsi %6, %c63 : index
      %intptr = memref.extract_aligned_pointer_as_index %alloca : memref<8xi8> -> index
      %8 = arith.index_cast %intptr : index to i64
      %9 = arith.index_cast %7 : index to i64
      %10 = arith.andi %8, %9 : i64
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
      %12 = llvm.load volatile %11 {alignment = 8 : i64} : !llvm.ptr -> i8
      %subview = memref.subview %arg0[%2] [1] [1] : memref<1xi8> to memref<1xi8, strided<[1], offset: ?>>
      %13 = arith.muli %thread_id_x, %arg5 : index
      %14 = arith.addi %arg5, %13 : index
      %c1_i64_5 = arith.constant 1 : i64
      %c1_6 = arith.constant 1 : index
      %15 = arith.subi %14, %c1_6 : index
      %16 = arith.index_cast %arg4 : i64 to index
      %17 = arith.subi %15, %16 : index
      %c63_i64_7 = arith.constant 63 : i64
      %c63_8 = arith.constant 63 : index
      %18 = arith.shrsi %17, %c63_8 : index
      %intptr_9 = memref.extract_aligned_pointer_as_index %alloca : memref<8xi8> -> index
      %19 = arith.index_cast %intptr_9 : index to i64
      %20 = arith.index_cast %18 : index to i64
      %21 = arith.andi %19, %20 : i64
      %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
      %23 = llvm.load volatile %22 {alignment = 8 : i64} : !llvm.ptr -> i8
      %subview_10 = memref.subview %arg3[%13] [1] [1] : memref<1xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_11 = arith.constant 0 : index
      %24 = arith.cmpi ult, %c0_11, %arg2 : index
      %c0_12 = arith.constant 0 : index
      %c0_i64 = arith.constant 0 : i64
      %25 = arith.index_cast %arg2 : index to i64
      %26 = arith.subi %c0_i64, %25 : i64
      %c63_i64_13 = arith.constant 63 : i64
      %27 = arith.shrsi %26, %c63_i64_13 : i64
      %intptr_14 = memref.extract_aligned_pointer_as_index %alloca : memref<8xi8> -> index
      %28 = arith.index_cast %intptr_14 : index to i64
      %29 = arith.andi %28, %27 : i64
      %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
      %31 = llvm.load volatile %30 {alignment = 8 : i64} : !llvm.ptr -> i8
      %subview_15 = memref.subview %subview[0] [1] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<1xi8, strided<[1], offset: ?>>
      %c0_16 = arith.constant 0 : index
      %32 = memref.load %subview_15[%c0_16] : memref<1xi8, strided<[1], offset: ?>>
      %c0_17 = arith.constant 0 : index
      %33 = arith.cmpi ult, %c0_17, %arg5 : index
      %c0_18 = arith.constant 0 : index
      %c0_i64_19 = arith.constant 0 : i64
      %34 = arith.index_cast %arg5 : index to i64
      %35 = arith.subi %c0_i64_19, %34 : i64
      %c63_i64_20 = arith.constant 63 : i64
      %36 = arith.shrsi %35, %c63_i64_20 : i64
      %intptr_21 = memref.extract_aligned_pointer_as_index %alloca : memref<8xi8> -> index
      %37 = arith.index_cast %intptr_21 : index to i64
      %38 = arith.andi %37, %36 : i64
      %39 = llvm.inttoptr %38 : i64 to !llvm.ptr
      %40 = llvm.load volatile %39 {alignment = 8 : i64} : !llvm.ptr -> i8
      %subview_22 = memref.subview %subview_10[0] [1] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<1xi8, strided<[1], offset: ?>>
      %c0_23 = arith.constant 0 : index
      memref.store %32, %subview_22[%c0_23] : memref<1xi8, strided<[1], offset: ?>>
      gpu.return
    }
    memref.global constant @memory_alloc_2 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %0 = memref.get_global @memory_alloc_2 : memref<11xi8>
  }
  module {
  }
  func.func private @_ZN3gpu12subslice_mut17hf19cfe966311ea83E(memref<1xi8>, i64, index, index) -> (memref<1xi8>, i64) attributes {gpu_codegen_builtin = "gpu.subslice_mut"}
  func.func private @_ZN3gpu8subslice17h92fa135b54d7a7b5E(memref<1xi8>, i64, index, index) -> (memref<1xi8>, i64) attributes {gpu_codegen_builtin = "gpu.subslice"}
  %c11 = arith.constant 11 : index
  func.func private @_ZN3gpu20add_mlir_string_attr17hb6fab0ade4590ffcE(memref<1xi8>, i64) -> index attributes {gpu_codegen_builtin = "add_mlir_string_attr"}
  %f = func.constant {gpu_codegen_builtin = "add_mlir_string_attr"} @_ZN3gpu20add_mlir_string_attr17hb6fab0ade4590ffcE : (memref<1xi8>, i64) -> index
  func.func private @_ZN3gpu9thread_id17h476e55ecb16338b5E() -> index attributes {gpu_codegen_builtin = "gpu.thread_id"}
  %f_0 = func.constant {gpu_codegen_builtin = "gpu.thread_id"} @_ZN3gpu9thread_id17h476e55ecb16338b5E : () -> index
  %f_1 = func.constant {gpu_codegen_builtin = "gpu.subslice"} @_ZN3gpu8subslice17h92fa135b54d7a7b5E : (memref<1xi8>, i64, index, index) -> (memref<1xi8>, i64)
  %f_2 = func.constant {gpu_codegen_builtin = "gpu.subslice_mut"} @_ZN3gpu12subslice_mut17hf19cfe966311ea83E : (memref<1xi8>, i64, index, index) -> (memref<1xi8>, i64)
  %c0 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
}
