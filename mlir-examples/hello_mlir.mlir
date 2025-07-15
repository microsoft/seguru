module attributes {gpu.container_module} {
  gpu.module @gpu attributes {visibility = "public"} {
    func.func private @_ZN3gpu3dim8block_id17hd40c964ff5c88ab7E(%arg0: i8) -> i64 {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_28 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_29 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %c0 = arith.constant 0 : index
      memref.store %arg0, %alloca_29[%c0] : memref<1xi8>
      %9 = arith.extui %arg0 : i8 to i64
      cf.switch %9 : i64, [
        default: ^bb1(%arg0 : i8),
        0: ^bb2(%arg0 : i8),
        1: ^bb3(%arg0 : i8),
        2: ^bb4(%arg0 : i8)
      ]
    ^bb1(%10: i8):  // pred: ^bb0
      cf.br ^bb6
    ^bb2(%11: i8):  // pred: ^bb0
      %12 = memref.get_global @memory_alloc_1 : memref<11xi8>
      %c11_30 = arith.constant 11 : index
      %block_id_x = gpu.block_id  x
      %c0_31 = arith.constant 0 : index
      %c0_32 = arith.constant 0 : index
      %view = memref.view %alloca[%c0_32][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_id_x, %view[%c0_31] : memref<1xindex>
      cf.br ^bb5(%11 : i8)
    ^bb3(%13: i8):  // pred: ^bb0
      %14 = memref.get_global @memory_alloc_2 : memref<11xi8>
      %c11_33 = arith.constant 11 : index
      %block_id_y = gpu.block_id  y
      %c0_34 = arith.constant 0 : index
      %c0_35 = arith.constant 0 : index
      %view_36 = memref.view %alloca[%c0_35][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_id_y, %view_36[%c0_34] : memref<1xindex>
      cf.br ^bb5(%13 : i8)
    ^bb4(%15: i8):  // pred: ^bb0
      %16 = memref.get_global @memory_alloc_3 : memref<11xi8>
      %c11_37 = arith.constant 11 : index
      %block_id_z = gpu.block_id  z
      %c0_38 = arith.constant 0 : index
      %c0_39 = arith.constant 0 : index
      %view_40 = memref.view %alloca[%c0_39][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_id_z, %view_40[%c0_38] : memref<1xindex>
      cf.br ^bb5(%15 : i8)
    ^bb5(%17: i8):  // 3 preds: ^bb2, ^bb3, ^bb4
      %c0_41 = arith.constant 0 : index
      %view_42 = memref.view %alloca[%c0_41][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      %c0_43 = arith.constant 0 : index
      %18 = memref.load %view_42[%c0_43] : memref<1xi64>
      return %18 : i64
    ^bb6:  // 2 preds: ^bb1, ^bb6
      %false = arith.constant false
      cf.assert %false, "unreachable"
      cf.br ^bb6
    }
    func.func private @_ZN3gpu3dim9block_dim17h409bcc0cefa30d66E(%arg0: i8) -> i64 {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_28 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_29 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %c0 = arith.constant 0 : index
      memref.store %arg0, %alloca_29[%c0] : memref<1xi8>
      %9 = arith.extui %arg0 : i8 to i64
      cf.switch %9 : i64, [
        default: ^bb1(%arg0 : i8),
        0: ^bb2(%arg0 : i8),
        1: ^bb3(%arg0 : i8),
        2: ^bb4(%arg0 : i8)
      ]
    ^bb1(%10: i8):  // pred: ^bb0
      cf.br ^bb6
    ^bb2(%11: i8):  // pred: ^bb0
      %12 = memref.get_global @memory_alloc_4 : memref<11xi8>
      %c11_30 = arith.constant 11 : index
      %block_dim_x = gpu.block_dim  x
      %c0_31 = arith.constant 0 : index
      %c0_32 = arith.constant 0 : index
      %view = memref.view %alloca[%c0_32][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_dim_x, %view[%c0_31] : memref<1xindex>
      cf.br ^bb5(%11 : i8)
    ^bb3(%13: i8):  // pred: ^bb0
      %14 = memref.get_global @memory_alloc_5 : memref<11xi8>
      %c11_33 = arith.constant 11 : index
      %block_dim_y = gpu.block_dim  y
      %c0_34 = arith.constant 0 : index
      %c0_35 = arith.constant 0 : index
      %view_36 = memref.view %alloca[%c0_35][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_dim_y, %view_36[%c0_34] : memref<1xindex>
      cf.br ^bb5(%13 : i8)
    ^bb4(%15: i8):  // pred: ^bb0
      %16 = memref.get_global @memory_alloc_6 : memref<11xi8>
      %c11_37 = arith.constant 11 : index
      %block_dim_z = gpu.block_dim  z
      %c0_38 = arith.constant 0 : index
      %c0_39 = arith.constant 0 : index
      %view_40 = memref.view %alloca[%c0_39][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_dim_z, %view_40[%c0_38] : memref<1xindex>
      cf.br ^bb5(%15 : i8)
    ^bb5(%17: i8):  // 3 preds: ^bb2, ^bb3, ^bb4
      %c0_41 = arith.constant 0 : index
      %view_42 = memref.view %alloca[%c0_41][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      %c0_43 = arith.constant 0 : index
      %18 = memref.load %view_42[%c0_43] : memref<1xi64>
      return %18 : i64
    ^bb6:  // 2 preds: ^bb1, ^bb6
      %false = arith.constant false
      cf.assert %false, "unreachable"
      cf.br ^bb6
    }
    func.func private @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E(%arg0: i8) -> i64 {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_28 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_29 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %c0 = arith.constant 0 : index
      memref.store %arg0, %alloca_29[%c0] : memref<1xi8>
      %9 = arith.extui %arg0 : i8 to i64
      cf.switch %9 : i64, [
        default: ^bb1(%arg0 : i8),
        0: ^bb2(%arg0 : i8),
        1: ^bb3(%arg0 : i8),
        2: ^bb4(%arg0 : i8)
      ]
    ^bb1(%10: i8):  // pred: ^bb0
      cf.br ^bb6
    ^bb2(%11: i8):  // pred: ^bb0
      %12 = memref.get_global @memory_alloc_7 : memref<11xi8>
      %c11_30 = arith.constant 11 : index
      %thread_id_x = gpu.thread_id  x
      %c0_31 = arith.constant 0 : index
      %c0_32 = arith.constant 0 : index
      %view = memref.view %alloca[%c0_32][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %thread_id_x, %view[%c0_31] : memref<1xindex>
      cf.br ^bb5(%11 : i8)
    ^bb3(%13: i8):  // pred: ^bb0
      %14 = memref.get_global @memory_alloc_8 : memref<11xi8>
      %c11_33 = arith.constant 11 : index
      %thread_id_y = gpu.thread_id  y
      %c0_34 = arith.constant 0 : index
      %c0_35 = arith.constant 0 : index
      %view_36 = memref.view %alloca[%c0_35][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %thread_id_y, %view_36[%c0_34] : memref<1xindex>
      cf.br ^bb5(%13 : i8)
    ^bb4(%15: i8):  // pred: ^bb0
      %16 = memref.get_global @memory_alloc_9 : memref<11xi8>
      %c11_37 = arith.constant 11 : index
      %thread_id_z = gpu.thread_id  z
      %c0_38 = arith.constant 0 : index
      %c0_39 = arith.constant 0 : index
      %view_40 = memref.view %alloca[%c0_39][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %thread_id_z, %view_40[%c0_38] : memref<1xindex>
      cf.br ^bb5(%15 : i8)
    ^bb5(%17: i8):  // 3 preds: ^bb2, ^bb3, ^bb4
      %c0_41 = arith.constant 0 : index
      %view_42 = memref.view %alloca[%c0_41][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      %c0_43 = arith.constant 0 : index
      %18 = memref.load %view_42[%c0_43] : memref<1xi64>
      return %18 : i64
    ^bb6:  // 2 preds: ^bb1, ^bb6
      %false = arith.constant false
      cf.assert %false, "unreachable"
      cf.br ^bb6
    }
    gpu.func @kernel_arith(%arg0: memref<4xi8>, %arg1: i64, %arg2: i64, %arg3: memref<4xi8>, %arg4: i64, %arg5: i64, %arg6: memref<4xi8>, %arg7: i64) kernel attributes {sym_visibility = "private"} {
      %alloca = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_28 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_29 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_30 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_31 = memref.alloca() {alignment = 4 : i64} : memref<4xi8>
      %alloca_32 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_33 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<4xi8> -> index
      %c0 = arith.constant 0 : index
      %c0_34 = arith.constant 0 : index
      %view = memref.view %alloca_33[%c0_34][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr, %view[%c0] : memref<1xindex>
      %c8_35 = arith.constant 8 : index
      %c1_i64 = arith.constant 1 : i64
      %c8_36 = arith.constant 8 : index
      %c1_i64_37 = arith.constant 1 : i64
      %c1 = arith.constant 1 : index
      %9 = arith.muli %c8_36, %c1 : index
      %subview = memref.subview %alloca_33[%9] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_38 = arith.constant 0 : index
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_39 = memref.view %reinterpret_cast[%offset][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg1, %view_39[%c0_38] : memref<1xi64>
      %alloca_40 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_41 = arith.constant 0 : index
      %c0_42 = arith.constant 0 : index
      %view_43 = memref.view %alloca_40[%c0_42][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %arg2, %view_43[%c0_41] : memref<1xi64>
      %alloca_44 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_45 = memref.extract_aligned_pointer_as_index %arg3 : memref<4xi8> -> index
      %c0_46 = arith.constant 0 : index
      %c0_47 = arith.constant 0 : index
      %view_48 = memref.view %alloca_44[%c0_47][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_45, %view_48[%c0_46] : memref<1xindex>
      %c8_49 = arith.constant 8 : index
      %c1_i64_50 = arith.constant 1 : i64
      %c8_51 = arith.constant 8 : index
      %c1_i64_52 = arith.constant 1 : i64
      %c1_53 = arith.constant 1 : index
      %10 = arith.muli %c8_51, %c1_53 : index
      %subview_54 = memref.subview %alloca_44[%10] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_55 = arith.constant 0 : index
      %base_buffer_56, %offset_57, %sizes_58, %strides_59 = memref.extract_strided_metadata %subview_54 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_60 = memref.reinterpret_cast %base_buffer_56 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_61 = memref.view %reinterpret_cast_60[%offset_57][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg4, %view_61[%c0_55] : memref<1xi64>
      %alloca_62 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_63 = arith.constant 0 : index
      %c0_64 = arith.constant 0 : index
      %view_65 = memref.view %alloca_62[%c0_64][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %arg5, %view_65[%c0_63] : memref<1xi64>
      %alloca_66 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_67 = memref.extract_aligned_pointer_as_index %arg6 : memref<4xi8> -> index
      %c0_68 = arith.constant 0 : index
      %c0_69 = arith.constant 0 : index
      %view_70 = memref.view %alloca_66[%c0_69][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_67, %view_70[%c0_68] : memref<1xindex>
      %c8_71 = arith.constant 8 : index
      %c1_i64_72 = arith.constant 1 : i64
      %c8_73 = arith.constant 8 : index
      %c1_i64_74 = arith.constant 1 : i64
      %c1_75 = arith.constant 1 : index
      %11 = arith.muli %c8_73, %c1_75 : index
      %subview_76 = memref.subview %alloca_66[%11] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_77 = arith.constant 0 : index
      %base_buffer_78, %offset_79, %sizes_80, %strides_81 = memref.extract_strided_metadata %subview_76 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_82 = memref.reinterpret_cast %base_buffer_78 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_83 = memref.view %reinterpret_cast_82[%offset_79][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg7, %view_83[%c0_77] : memref<1xi64>
      %c0_i8_84 = arith.constant 0 : i8
      %c0_85 = arith.constant 0 : index
      memref.store %c0_i8_84, %alloca[%c0_85] : memref<1xi8>
      %c0_86 = arith.constant 0 : index
      %12 = memref.load %alloca[%c0_86] : memref<1xi8>
      %13 = func.call @_ZN3gpu3dim9block_dim17h409bcc0cefa30d66E(%12) : (i8) -> i64
      %c0_i8_87 = arith.constant 0 : i8
      %c0_88 = arith.constant 0 : index
      memref.store %c0_i8_87, %alloca_28[%c0_88] : memref<1xi8>
      %c0_89 = arith.constant 0 : index
      %14 = memref.load %alloca_28[%c0_89] : memref<1xi8>
      %15 = func.call @_ZN3gpu3dim8block_id17hd40c964ff5c88ab7E(%14) : (i8) -> i64
      %16 = arith.muli %13, %15 : i64
      %false = arith.constant false
      %c0_i8_90 = arith.constant 0 : i8
      %c0_91 = arith.constant 0 : index
      memref.store %c0_i8_90, %alloca_29[%c0_91] : memref<1xi8>
      %c0_92 = arith.constant 0 : index
      %17 = memref.load %alloca_29[%c0_92] : memref<1xi8>
      %18 = func.call @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E(%17) : (i8) -> i64
      %19 = arith.addi %16, %18 : i64
      %false_93 = arith.constant false
      %alloca_94 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_95 = arith.constant 0 : index
      %c0_96 = arith.constant 0 : index
      %view_97 = memref.view %alloca_94[%c0_96][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %19, %view_97[%c0_95] : memref<1xi64>
      %20 = arith.muli %19, %arg5 : i64
      %false_98 = arith.constant false
      %21 = arith.addi %arg5, %20 : i64
      %c1_i64_99 = arith.constant 1 : i64
      %22 = arith.subi %21, %c1_i64_99 : i64
      %23 = arith.subi %22, %arg4 : i64
      %c63_i64 = arith.constant 63 : i64
      %24 = arith.shrsi %23, %c63_i64 : i64
      %intptr_100 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %25 = arith.index_cast %intptr_100 : index to i64
      %26 = arith.andi %25, %24 : i64
      %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
      %28 = llvm.load volatile %27 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4 = arith.constant 4 : index
      %c4_101 = arith.constant 4 : index
      %c4_i64 = arith.constant 4 : i64
      %29 = arith.muli %20, %c4_i64 : i64
      %c4_i64_102 = arith.constant 4 : i64
      %c4_i64_103 = arith.constant 4 : i64
      %30 = arith.muli %20, %c4_i64_103 : i64
      %31 = arith.index_cast %30 : i64 to index
      %subview_104 = memref.subview %arg3[%31] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %alloca_105 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_106 = memref.extract_aligned_pointer_as_index %subview_104 : memref<4xi8, strided<[1], offset: ?>> -> index
      %c0_107 = arith.constant 0 : index
      %c0_108 = arith.constant 0 : index
      %view_109 = memref.view %alloca_105[%c0_108][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_106, %view_109[%c0_107] : memref<1xindex>
      %c8_110 = arith.constant 8 : index
      %c1_i64_111 = arith.constant 1 : i64
      %c8_112 = arith.constant 8 : index
      %c1_i64_113 = arith.constant 1 : i64
      %c1_114 = arith.constant 1 : index
      %32 = arith.muli %c8_112, %c1_114 : index
      %subview_115 = memref.subview %alloca_105[%32] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_116 = arith.constant 0 : index
      %base_buffer_117, %offset_118, %sizes_119, %strides_120 = memref.extract_strided_metadata %subview_115 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_121 = memref.reinterpret_cast %base_buffer_117 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_122 = memref.view %reinterpret_cast_121[%offset_118][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg5, %view_122[%c0_116] : memref<1xi64>
      %33 = arith.muli %19, %arg2 : i64
      %false_123 = arith.constant false
      %34 = arith.addi %arg2, %33 : i64
      %c1_i64_124 = arith.constant 1 : i64
      %35 = arith.subi %34, %c1_i64_124 : i64
      %36 = arith.subi %35, %arg1 : i64
      %c63_i64_125 = arith.constant 63 : i64
      %37 = arith.shrsi %36, %c63_i64_125 : i64
      %intptr_126 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %38 = arith.index_cast %intptr_126 : index to i64
      %39 = arith.andi %38, %37 : i64
      %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
      %41 = llvm.load volatile %40 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4_127 = arith.constant 4 : index
      %c4_128 = arith.constant 4 : index
      %c4_i64_129 = arith.constant 4 : i64
      %42 = arith.muli %33, %c4_i64_129 : i64
      %c4_i64_130 = arith.constant 4 : i64
      %c4_i64_131 = arith.constant 4 : i64
      %43 = arith.muli %33, %c4_i64_131 : i64
      %44 = arith.index_cast %43 : i64 to index
      %subview_132 = memref.subview %arg0[%44] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %alloca_133 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_134 = memref.extract_aligned_pointer_as_index %subview_132 : memref<4xi8, strided<[1], offset: ?>> -> index
      %c0_135 = arith.constant 0 : index
      %c0_136 = arith.constant 0 : index
      %view_137 = memref.view %alloca_133[%c0_136][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_134, %view_137[%c0_135] : memref<1xindex>
      %c8_138 = arith.constant 8 : index
      %c1_i64_139 = arith.constant 1 : i64
      %c8_140 = arith.constant 8 : index
      %c1_i64_141 = arith.constant 1 : i64
      %c1_142 = arith.constant 1 : index
      %45 = arith.muli %c8_140, %c1_142 : index
      %subview_143 = memref.subview %alloca_133[%45] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_144 = arith.constant 0 : index
      %base_buffer_145, %offset_146, %sizes_147, %strides_148 = memref.extract_strided_metadata %subview_143 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_149 = memref.reinterpret_cast %base_buffer_145 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_150 = memref.view %reinterpret_cast_149[%offset_146][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg2, %view_150[%c0_144] : memref<1xi64>
      %c0_i8_151 = arith.constant 0 : i8
      %c0_152 = arith.constant 0 : index
      memref.store %c0_i8_151, %alloca_30[%c0_152] : memref<1xi8>
      %c0_153 = arith.constant 0 : index
      %46 = memref.load %alloca_30[%c0_153] : memref<1xi8>
      %47 = func.call @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E(%46) : (i8) -> i64
      %alloca_154 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_155 = arith.constant 0 : index
      %c0_156 = arith.constant 0 : index
      %view_157 = memref.view %alloca_154[%c0_156][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %47, %view_157[%c0_155] : memref<1xi64>
      %c0_i64_158 = arith.constant 0 : i64
      %c0_i64_159 = arith.constant 0 : i64
      %48 = arith.cmpi ult, %c0_i64_159, %arg2 : i64
      %49 = arith.subi %c0_i64_158, %arg2 : i64
      %c63_i64_160 = arith.constant 63 : i64
      %50 = arith.shrsi %49, %c63_i64_160 : i64
      %intptr_161 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %51 = arith.index_cast %intptr_161 : index to i64
      %52 = arith.andi %51, %50 : i64
      %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
      %54 = llvm.load volatile %53 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_162 = arith.constant 0 : i64
      %c4_i64_163 = arith.constant 4 : i64
      %c0_i64_164 = arith.constant 0 : i64
      %c4_i64_165 = arith.constant 4 : i64
      %55 = arith.muli %c0_i64_164, %c4_i64_165 : i64
      %56 = arith.index_cast %55 : i64 to index
      %base_buffer_166, %offset_167, %sizes_168, %strides_169 = memref.extract_strided_metadata %subview_132 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_170 = memref.reinterpret_cast %base_buffer_166 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_171 = memref.view %reinterpret_cast_170[%offset_167][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_172 = memref.subview %view_171[%56] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %base_buffer_173, %offset_174, %sizes_175, %strides_176 = memref.extract_strided_metadata %subview_172 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_177 = memref.reinterpret_cast %base_buffer_173 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_178 = memref.view %reinterpret_cast_177[%offset_174][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %c0_179 = arith.constant 0 : index
      %57 = memref.load %view_178[%c0_179] : memref<1xi32>
      %58 = arith.cmpi ult, %47, %arg7 : i64
      %59 = arith.subi %47, %arg7 : i64
      %c63_i64_180 = arith.constant 63 : i64
      %60 = arith.shrsi %59, %c63_i64_180 : i64
      %intptr_181 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %61 = arith.index_cast %intptr_181 : index to i64
      %62 = arith.andi %61, %60 : i64
      %63 = llvm.inttoptr %62 : i64 to !llvm.ptr
      %64 = llvm.load volatile %63 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4_i64_182 = arith.constant 4 : i64
      %c4_i64_183 = arith.constant 4 : i64
      %65 = arith.muli %47, %c4_i64_183 : i64
      %66 = arith.index_cast %65 : i64 to index
      %subview_184 = memref.subview %arg6[%66] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %base_buffer_185, %offset_186, %sizes_187, %strides_188 = memref.extract_strided_metadata %subview_184 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_189 = memref.reinterpret_cast %base_buffer_185 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_190 = memref.view %reinterpret_cast_189[%offset_186][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %c0_191 = arith.constant 0 : index
      %67 = memref.load %view_190[%c0_191] : memref<1xi32>
      %68 = arith.addi %57, %67 : i32
      %false_192 = arith.constant false
      %c0_i64_193 = arith.constant 0 : i64
      %c0_i64_194 = arith.constant 0 : i64
      %69 = arith.cmpi ult, %c0_i64_194, %arg5 : i64
      %70 = arith.subi %c0_i64_193, %arg5 : i64
      %c63_i64_195 = arith.constant 63 : i64
      %71 = arith.shrsi %70, %c63_i64_195 : i64
      %intptr_196 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %72 = arith.index_cast %intptr_196 : index to i64
      %73 = arith.andi %72, %71 : i64
      %74 = llvm.inttoptr %73 : i64 to !llvm.ptr
      %75 = llvm.load volatile %74 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_197 = arith.constant 0 : i64
      %c4_i64_198 = arith.constant 4 : i64
      %c0_i64_199 = arith.constant 0 : i64
      %c4_i64_200 = arith.constant 4 : i64
      %76 = arith.muli %c0_i64_199, %c4_i64_200 : i64
      %77 = arith.index_cast %76 : i64 to index
      %base_buffer_201, %offset_202, %sizes_203, %strides_204 = memref.extract_strided_metadata %subview_104 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_205 = memref.reinterpret_cast %base_buffer_201 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_206 = memref.view %reinterpret_cast_205[%offset_202][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_207 = memref.subview %view_206[%77] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %c0_208 = arith.constant 0 : index
      %base_buffer_209, %offset_210, %sizes_211, %strides_212 = memref.extract_strided_metadata %subview_207 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_213 = memref.reinterpret_cast %base_buffer_209 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_214 = memref.view %reinterpret_cast_213[%offset_210][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      memref.store %68, %view_214[%c0_208] : memref<1xi32>
      %c0_i64_215 = arith.constant 0 : i64
      %c0_i64_216 = arith.constant 0 : i64
      %78 = arith.cmpi ult, %c0_i64_216, %arg5 : i64
      %79 = arith.subi %c0_i64_215, %arg5 : i64
      %c63_i64_217 = arith.constant 63 : i64
      %80 = arith.shrsi %79, %c63_i64_217 : i64
      %intptr_218 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %81 = arith.index_cast %intptr_218 : index to i64
      %82 = arith.andi %81, %80 : i64
      %83 = llvm.inttoptr %82 : i64 to !llvm.ptr
      %84 = llvm.load volatile %83 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_219 = arith.constant 0 : i64
      %c4_i64_220 = arith.constant 4 : i64
      %c0_i64_221 = arith.constant 0 : i64
      %c4_i64_222 = arith.constant 4 : i64
      %85 = arith.muli %c0_i64_221, %c4_i64_222 : i64
      %86 = arith.index_cast %85 : i64 to index
      %base_buffer_223, %offset_224, %sizes_225, %strides_226 = memref.extract_strided_metadata %subview_104 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_227 = memref.reinterpret_cast %base_buffer_223 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_228 = memref.view %reinterpret_cast_227[%offset_224][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_229 = memref.subview %view_228[%86] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_230 = arith.constant 1 : i32
      %c0_231 = arith.constant 0 : index
      %base_buffer_232, %offset_233, %sizes_234, %strides_235 = memref.extract_strided_metadata %subview_229 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_236 = memref.reinterpret_cast %base_buffer_232 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_237 = memref.view %reinterpret_cast_236[%offset_233][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %87 = memref.atomic_rmw addi %c1_i32_230, %view_237[%c0_231] : (i32, memref<1xi32>) -> i32
      %c0_i64_238 = arith.constant 0 : i64
      %c0_i64_239 = arith.constant 0 : i64
      %88 = arith.cmpi ult, %c0_i64_239, %arg5 : i64
      %89 = arith.subi %c0_i64_238, %arg5 : i64
      %c63_i64_240 = arith.constant 63 : i64
      %90 = arith.shrsi %89, %c63_i64_240 : i64
      %intptr_241 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %91 = arith.index_cast %intptr_241 : index to i64
      %92 = arith.andi %91, %90 : i64
      %93 = llvm.inttoptr %92 : i64 to !llvm.ptr
      %94 = llvm.load volatile %93 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_242 = arith.constant 0 : i64
      %c4_i64_243 = arith.constant 4 : i64
      %c0_i64_244 = arith.constant 0 : i64
      %c4_i64_245 = arith.constant 4 : i64
      %95 = arith.muli %c0_i64_244, %c4_i64_245 : i64
      %96 = arith.index_cast %95 : i64 to index
      %base_buffer_246, %offset_247, %sizes_248, %strides_249 = memref.extract_strided_metadata %subview_104 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_250 = memref.reinterpret_cast %base_buffer_246 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_251 = memref.view %reinterpret_cast_250[%offset_247][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_252 = memref.subview %view_251[%96] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %base_buffer_253, %offset_254, %sizes_255, %strides_256 = memref.extract_strided_metadata %subview_252 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_257 = memref.reinterpret_cast %base_buffer_253 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_258 = memref.view %reinterpret_cast_257[%offset_254][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %c0_259 = arith.constant 0 : index
      %97 = memref.load %view_258[%c0_259] : memref<1xi32>
      %c30_i32 = arith.constant 30 : i32
      %c30_i32_260 = arith.constant 30 : i32
      %98 = arith.addi %97, %c30_i32_260 : i32
      %false_261 = arith.constant false
      %alloca_262 = memref.alloca() {alignment = 4 : i64} : memref<4xi8>
      %c0_263 = arith.constant 0 : index
      %c0_264 = arith.constant 0 : index
      %view_265 = memref.view %alloca_262[%c0_264][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xi32>
      memref.store %98, %view_265[%c0_263] : memref<1xi32>
      %99 = llvm.inline_asm has_side_effects is_align_stack "mov.u32 ${0}, ${1};", "=&r,r,~{memory}" %98 : (i32) -> i32
      %c0_266 = arith.constant 0 : index
      %c0_267 = arith.constant 0 : index
      %view_268 = memref.view %alloca_31[%c0_267][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xi32>
      memref.store %99, %view_268[%c0_266] : memref<1xi32>
      %c0_269 = arith.constant 0 : index
      %view_270 = memref.view %alloca_31[%c0_269][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xi32>
      %c0_271 = arith.constant 0 : index
      %100 = memref.load %view_270[%c0_271] : memref<1xi32>
      %c0_i64_272 = arith.constant 0 : i64
      %c0_i64_273 = arith.constant 0 : i64
      %101 = arith.cmpi ult, %c0_i64_273, %arg5 : i64
      %102 = arith.subi %c0_i64_272, %arg5 : i64
      %c63_i64_274 = arith.constant 63 : i64
      %103 = arith.shrsi %102, %c63_i64_274 : i64
      %intptr_275 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<8xi8> -> index
      %104 = arith.index_cast %intptr_275 : index to i64
      %105 = arith.andi %104, %103 : i64
      %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
      %107 = llvm.load volatile %106 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_276 = arith.constant 0 : i64
      %c4_i64_277 = arith.constant 4 : i64
      %c0_i64_278 = arith.constant 0 : i64
      %c4_i64_279 = arith.constant 4 : i64
      %108 = arith.muli %c0_i64_278, %c4_i64_279 : i64
      %109 = arith.index_cast %108 : i64 to index
      %base_buffer_280, %offset_281, %sizes_282, %strides_283 = memref.extract_strided_metadata %subview_104 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_284 = memref.reinterpret_cast %base_buffer_280 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_285 = memref.view %reinterpret_cast_284[%offset_281][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_286 = memref.subview %view_285[%109] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %c0_287 = arith.constant 0 : index
      %base_buffer_288, %offset_289, %sizes_290, %strides_291 = memref.extract_strided_metadata %subview_286 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_292 = memref.reinterpret_cast %base_buffer_288 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_293 = memref.view %reinterpret_cast_292[%offset_289][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      memref.store %100, %view_293[%c0_287] : memref<1xi32>
      gpu.return
    }
    memref.global constant @memory_alloc_1 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %0 = memref.get_global @memory_alloc_1 : memref<11xi8>
    func.func private @_ZN3gpu20add_mlir_string_attr17h0b16585ac8dd7908E(memref<1xi8>, i64) -> i64 attributes {gpu_codegen_builtin = "gpu::add_mlir_string_attr"}
    %f_21 = func.constant {gpu_codegen_builtin = "gpu::add_mlir_string_attr"} @_ZN3gpu20add_mlir_string_attr17h0b16585ac8dd7908E : (memref<1xi8>, i64) -> i64
    func.func private @_ZN3gpu3dim9_block_id17h9a3bf55096aa3164E() -> i64 attributes {gpu_codegen_builtin = "gpu::block_id"}
    %f_22 = func.constant {gpu_codegen_builtin = "gpu::block_id"} @_ZN3gpu3dim9_block_id17h9a3bf55096aa3164E : () -> i64
    memref.global constant @memory_alloc_2 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 121, 62]>
    %1 = memref.get_global @memory_alloc_2 : memref<11xi8>
    memref.global constant @memory_alloc_3 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 122, 62]>
    %2 = memref.get_global @memory_alloc_3 : memref<11xi8>
    memref.global constant @memory_alloc_4 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %3 = memref.get_global @memory_alloc_4 : memref<11xi8>
    func.func private @_ZN3gpu3dim10_block_dim17h8a4ce5d49d624b14E() -> i64 attributes {gpu_codegen_builtin = "gpu::block_dim"}
    %f_23 = func.constant {gpu_codegen_builtin = "gpu::block_dim"} @_ZN3gpu3dim10_block_dim17h8a4ce5d49d624b14E : () -> i64
    memref.global constant @memory_alloc_5 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 121, 62]>
    %4 = memref.get_global @memory_alloc_5 : memref<11xi8>
    memref.global constant @memory_alloc_6 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 122, 62]>
    %5 = memref.get_global @memory_alloc_6 : memref<11xi8>
    memref.global constant @memory_alloc_7 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %6 = memref.get_global @memory_alloc_7 : memref<11xi8>
    func.func private @_ZN3gpu3dim10_thread_id17h1829fde4294be6f3E() -> i64 attributes {gpu_codegen_builtin = "gpu::thread_id"}
    %f_24 = func.constant {gpu_codegen_builtin = "gpu::thread_id"} @_ZN3gpu3dim10_thread_id17h1829fde4294be6f3E : () -> i64
    memref.global constant @memory_alloc_8 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 121, 62]>
    %7 = memref.get_global @memory_alloc_8 : memref<11xi8>
    memref.global constant @memory_alloc_9 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 122, 62]>
    %8 = memref.get_global @memory_alloc_9 : memref<11xi8>
    %f_25 = func.constant @_ZN3gpu3dim9block_dim17h409bcc0cefa30d66E : (i8) -> i64
    %f_26 = func.constant @_ZN3gpu3dim8block_id17hd40c964ff5c88ab7E : (i8) -> i64
    %f_27 = func.constant @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E : (i8) -> i64
  }
  module {
  }
  func.func private @_ZN3gpu10atomic_add17h3a9c21a4f17e3e4bE(memref<4xi8>, i32) -> i32 attributes {gpu_codegen_builtin = "gpu::atomic_add"}
  func.func private @_ZN3gpu12subslice_mut17h8501f848cdb8fb24E(memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64) attributes {gpu_codegen_builtin = "gpu::subslice_mut"}
  func.func private @_ZN3gpu8subslice17h1cbe85c5a0ff6edfE(memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64) attributes {gpu_codegen_builtin = "gpu::subslice"}
  %c11 = arith.constant 11 : index
  %c11_0 = arith.constant 11 : index
  %c11_1 = arith.constant 11 : index
  %c11_2 = arith.constant 11 : index
  %c11_3 = arith.constant 11 : index
  %c11_4 = arith.constant 11 : index
  %c11_5 = arith.constant 11 : index
  %c11_6 = arith.constant 11 : index
  %c11_7 = arith.constant 11 : index
  %c0_i64 = arith.constant 0 : i64
  %c8 = arith.constant 8 : index
  %c0_i64_8 = arith.constant 0 : i64
  %c8_9 = arith.constant 8 : index
  %c0_i64_10 = arith.constant 0 : i64
  %c8_11 = arith.constant 8 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i8_12 = arith.constant 0 : i8
  %c0_i8_13 = arith.constant 0 : i8
  %f = func.constant {gpu_codegen_builtin = "gpu::subslice_mut"} @_ZN3gpu12subslice_mut17h8501f848cdb8fb24E : (memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64)
  %c0_i64_14 = arith.constant 0 : i64
  %c8_15 = arith.constant 8 : index
  %f_16 = func.constant {gpu_codegen_builtin = "gpu::subslice"} @_ZN3gpu8subslice17h1cbe85c5a0ff6edfE : (memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64)
  %c0_i64_17 = arith.constant 0 : i64
  %c8_18 = arith.constant 8 : index
  %c0_i8_19 = arith.constant 0 : i8
  %f_20 = func.constant {gpu_codegen_builtin = "gpu::atomic_add"} @_ZN3gpu10atomic_add17h3a9c21a4f17e3e4bE : (memref<4xi8>, i32) -> i32
}
