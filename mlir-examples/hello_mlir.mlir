module attributes {gpu.container_module} {
  gpu.module @gpu attributes {visibility = "public"} {
    func.func private @_ZN3gpu10__ldcs_f3217h82bc33ad1a5bfb48E(%arg0: memref<4xi8>) -> f32 {
      %alloca = memref.alloca() {alignment = 4 : i64} : memref<4xi8>
      %alloca_39 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_40 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<4xi8> -> index
      %c0 = arith.constant 0 : index
      %c0_41 = arith.constant 0 : index
      %view = memref.view %alloca_40[%c0_41][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %intptr, %view[%c0] : memref<1xindex>
      %alloca_42 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %intptr_43 = memref.extract_aligned_pointer_as_index %arg0 : memref<4xi8> -> index
      %c0_44 = arith.constant 0 : index
      %c0_45 = arith.constant 0 : index
      %view_46 = memref.view %alloca_42[%c0_45][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %intptr_43, %view_46[%c0_44] : memref<1xindex>
      %intptr_47 = memref.extract_aligned_pointer_as_index %arg0 : memref<4xi8> -> index
      %9 = arith.index_cast %intptr_47 : index to i64
      %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
      %11 = llvm.inline_asm has_side_effects is_align_stack "ld.global.cs.f32 ${0}, [${1}];", "=&r,l,~{memory}" %10 : (!llvm.ptr) -> f32
      %c0_48 = arith.constant 0 : index
      %c0_49 = arith.constant 0 : index
      %view_50 = memref.view %alloca[%c0_49][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xf32>
      memref.store %11, %view_50[%c0_48] : memref<1xf32>
      %c0_51 = arith.constant 0 : index
      %view_52 = memref.view %alloca[%c0_51][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xf32>
      %c0_53 = arith.constant 0 : index
      %12 = memref.load %view_52[%c0_53] : memref<1xf32>
      return %12 : f32
    }
    func.func private @_ZN3gpu3dim8block_id17hd40c964ff5c88ab7E(%arg0: i8) -> i64 {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_39 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_40 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %c0 = arith.constant 0 : index
      memref.store %arg0, %alloca_40[%c0] : memref<1xi8>
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
      %c11_41 = arith.constant 11 : index
      %block_id_x = gpu.block_id  x
      %c0_42 = arith.constant 0 : index
      %c0_43 = arith.constant 0 : index
      %view = memref.view %alloca[%c0_43][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_id_x, %view[%c0_42] : memref<1xindex>
      cf.br ^bb5(%11 : i8)
    ^bb3(%13: i8):  // pred: ^bb0
      %14 = memref.get_global @memory_alloc_2 : memref<11xi8>
      %c11_44 = arith.constant 11 : index
      %block_id_y = gpu.block_id  y
      %c0_45 = arith.constant 0 : index
      %c0_46 = arith.constant 0 : index
      %view_47 = memref.view %alloca[%c0_46][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_id_y, %view_47[%c0_45] : memref<1xindex>
      cf.br ^bb5(%13 : i8)
    ^bb4(%15: i8):  // pred: ^bb0
      %16 = memref.get_global @memory_alloc_3 : memref<11xi8>
      %c11_48 = arith.constant 11 : index
      %block_id_z = gpu.block_id  z
      %c0_49 = arith.constant 0 : index
      %c0_50 = arith.constant 0 : index
      %view_51 = memref.view %alloca[%c0_50][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_id_z, %view_51[%c0_49] : memref<1xindex>
      cf.br ^bb5(%15 : i8)
    ^bb5(%17: i8):  // 3 preds: ^bb2, ^bb3, ^bb4
      %c0_52 = arith.constant 0 : index
      %view_53 = memref.view %alloca[%c0_52][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      %c0_54 = arith.constant 0 : index
      %18 = memref.load %view_53[%c0_54] : memref<1xi64>
      return %18 : i64
    ^bb6:  // 2 preds: ^bb1, ^bb6
      %false = arith.constant false
      cf.assert %false, "unreachable"
      cf.br ^bb6
    }
    func.func private @_ZN3gpu3dim9block_dim17h409bcc0cefa30d66E(%arg0: i8) -> i64 {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_39 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_40 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %c0 = arith.constant 0 : index
      memref.store %arg0, %alloca_40[%c0] : memref<1xi8>
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
      %c11_41 = arith.constant 11 : index
      %block_dim_x = gpu.block_dim  x
      %c0_42 = arith.constant 0 : index
      %c0_43 = arith.constant 0 : index
      %view = memref.view %alloca[%c0_43][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_dim_x, %view[%c0_42] : memref<1xindex>
      cf.br ^bb5(%11 : i8)
    ^bb3(%13: i8):  // pred: ^bb0
      %14 = memref.get_global @memory_alloc_5 : memref<11xi8>
      %c11_44 = arith.constant 11 : index
      %block_dim_y = gpu.block_dim  y
      %c0_45 = arith.constant 0 : index
      %c0_46 = arith.constant 0 : index
      %view_47 = memref.view %alloca[%c0_46][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_dim_y, %view_47[%c0_45] : memref<1xindex>
      cf.br ^bb5(%13 : i8)
    ^bb4(%15: i8):  // pred: ^bb0
      %16 = memref.get_global @memory_alloc_6 : memref<11xi8>
      %c11_48 = arith.constant 11 : index
      %block_dim_z = gpu.block_dim  z
      %c0_49 = arith.constant 0 : index
      %c0_50 = arith.constant 0 : index
      %view_51 = memref.view %alloca[%c0_50][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %block_dim_z, %view_51[%c0_49] : memref<1xindex>
      cf.br ^bb5(%15 : i8)
    ^bb5(%17: i8):  // 3 preds: ^bb2, ^bb3, ^bb4
      %c0_52 = arith.constant 0 : index
      %view_53 = memref.view %alloca[%c0_52][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      %c0_54 = arith.constant 0 : index
      %18 = memref.load %view_53[%c0_54] : memref<1xi64>
      return %18 : i64
    ^bb6:  // 2 preds: ^bb1, ^bb6
      %false = arith.constant false
      cf.assert %false, "unreachable"
      cf.br ^bb6
    }
    func.func private @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E(%arg0: i8) -> i64 {
      %alloca = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_39 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_40 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %c0 = arith.constant 0 : index
      memref.store %arg0, %alloca_40[%c0] : memref<1xi8>
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
      %c11_41 = arith.constant 11 : index
      %thread_id_x = gpu.thread_id  x
      %c0_42 = arith.constant 0 : index
      %c0_43 = arith.constant 0 : index
      %view = memref.view %alloca[%c0_43][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %thread_id_x, %view[%c0_42] : memref<1xindex>
      cf.br ^bb5(%11 : i8)
    ^bb3(%13: i8):  // pred: ^bb0
      %14 = memref.get_global @memory_alloc_8 : memref<11xi8>
      %c11_44 = arith.constant 11 : index
      %thread_id_y = gpu.thread_id  y
      %c0_45 = arith.constant 0 : index
      %c0_46 = arith.constant 0 : index
      %view_47 = memref.view %alloca[%c0_46][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %thread_id_y, %view_47[%c0_45] : memref<1xindex>
      cf.br ^bb5(%13 : i8)
    ^bb4(%15: i8):  // pred: ^bb0
      %16 = memref.get_global @memory_alloc_9 : memref<11xi8>
      %c11_48 = arith.constant 11 : index
      %thread_id_z = gpu.thread_id  z
      %c0_49 = arith.constant 0 : index
      %c0_50 = arith.constant 0 : index
      %view_51 = memref.view %alloca[%c0_50][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xindex>
      memref.store %thread_id_z, %view_51[%c0_49] : memref<1xindex>
      cf.br ^bb5(%15 : i8)
    ^bb5(%17: i8):  // 3 preds: ^bb2, ^bb3, ^bb4
      %c0_52 = arith.constant 0 : index
      %view_53 = memref.view %alloca[%c0_52][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      %c0_54 = arith.constant 0 : index
      %18 = memref.load %view_53[%c0_54] : memref<1xi64>
      return %18 : i64
    ^bb6:  // 2 preds: ^bb1, ^bb6
      %false = arith.constant false
      cf.assert %false, "unreachable"
      cf.br ^bb6
    }
    gpu.func @kernel_arith(%arg0: memref<4xi8>, %arg1: i64, %arg2: i64, %arg3: memref<4xi8>, %arg4: i64, %arg5: i64, %arg6: memref<4xi8>, %arg7: i64, %arg8: memref<4xi8>, %arg9: i64, %arg10: i64, %arg11: memref<4xi8>, %arg12: i64) kernel attributes {sym_visibility = "private"} {
      %alloca = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_39 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_40 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_41 = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
      %alloca_42 = memref.alloca() {alignment = 4 : i64} : memref<4xi8>
      %alloca_43 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %alloca_44 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<4xi8> -> index
      %c0 = arith.constant 0 : index
      %c0_45 = arith.constant 0 : index
      %view = memref.view %alloca_44[%c0_45][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr, %view[%c0] : memref<1xindex>
      %c8_46 = arith.constant 8 : index
      %c1_i64 = arith.constant 1 : i64
      %c8_47 = arith.constant 8 : index
      %c1_i64_48 = arith.constant 1 : i64
      %c1 = arith.constant 1 : index
      %9 = arith.muli %c8_47, %c1 : index
      %subview = memref.subview %alloca_44[%9] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_49 = arith.constant 0 : index
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_50 = memref.view %reinterpret_cast[%offset][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg1, %view_50[%c0_49] : memref<1xi64>
      %alloca_51 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_52 = arith.constant 0 : index
      %c0_53 = arith.constant 0 : index
      %view_54 = memref.view %alloca_51[%c0_53][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %arg2, %view_54[%c0_52] : memref<1xi64>
      %alloca_55 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_56 = memref.extract_aligned_pointer_as_index %arg3 : memref<4xi8> -> index
      %c0_57 = arith.constant 0 : index
      %c0_58 = arith.constant 0 : index
      %view_59 = memref.view %alloca_55[%c0_58][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_56, %view_59[%c0_57] : memref<1xindex>
      %c8_60 = arith.constant 8 : index
      %c1_i64_61 = arith.constant 1 : i64
      %c8_62 = arith.constant 8 : index
      %c1_i64_63 = arith.constant 1 : i64
      %c1_64 = arith.constant 1 : index
      %10 = arith.muli %c8_62, %c1_64 : index
      %subview_65 = memref.subview %alloca_55[%10] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_66 = arith.constant 0 : index
      %base_buffer_67, %offset_68, %sizes_69, %strides_70 = memref.extract_strided_metadata %subview_65 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_71 = memref.reinterpret_cast %base_buffer_67 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_72 = memref.view %reinterpret_cast_71[%offset_68][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg4, %view_72[%c0_66] : memref<1xi64>
      %alloca_73 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_74 = arith.constant 0 : index
      %c0_75 = arith.constant 0 : index
      %view_76 = memref.view %alloca_73[%c0_75][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %arg5, %view_76[%c0_74] : memref<1xi64>
      %alloca_77 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_78 = memref.extract_aligned_pointer_as_index %arg6 : memref<4xi8> -> index
      %c0_79 = arith.constant 0 : index
      %c0_80 = arith.constant 0 : index
      %view_81 = memref.view %alloca_77[%c0_80][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_78, %view_81[%c0_79] : memref<1xindex>
      %c8_82 = arith.constant 8 : index
      %c1_i64_83 = arith.constant 1 : i64
      %c8_84 = arith.constant 8 : index
      %c1_i64_85 = arith.constant 1 : i64
      %c1_86 = arith.constant 1 : index
      %11 = arith.muli %c8_84, %c1_86 : index
      %subview_87 = memref.subview %alloca_77[%11] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_88 = arith.constant 0 : index
      %base_buffer_89, %offset_90, %sizes_91, %strides_92 = memref.extract_strided_metadata %subview_87 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_93 = memref.reinterpret_cast %base_buffer_89 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_94 = memref.view %reinterpret_cast_93[%offset_90][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg7, %view_94[%c0_88] : memref<1xi64>
      %alloca_95 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_96 = memref.extract_aligned_pointer_as_index %arg8 : memref<4xi8> -> index
      %c0_97 = arith.constant 0 : index
      %c0_98 = arith.constant 0 : index
      %view_99 = memref.view %alloca_95[%c0_98][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_96, %view_99[%c0_97] : memref<1xindex>
      %c8_100 = arith.constant 8 : index
      %c1_i64_101 = arith.constant 1 : i64
      %c8_102 = arith.constant 8 : index
      %c1_i64_103 = arith.constant 1 : i64
      %c1_104 = arith.constant 1 : index
      %12 = arith.muli %c8_102, %c1_104 : index
      %subview_105 = memref.subview %alloca_95[%12] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_106 = arith.constant 0 : index
      %base_buffer_107, %offset_108, %sizes_109, %strides_110 = memref.extract_strided_metadata %subview_105 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_111 = memref.reinterpret_cast %base_buffer_107 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_112 = memref.view %reinterpret_cast_111[%offset_108][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg9, %view_112[%c0_106] : memref<1xi64>
      %alloca_113 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_114 = arith.constant 0 : index
      %c0_115 = arith.constant 0 : index
      %view_116 = memref.view %alloca_113[%c0_115][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %arg10, %view_116[%c0_114] : memref<1xi64>
      %alloca_117 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_118 = memref.extract_aligned_pointer_as_index %arg11 : memref<4xi8> -> index
      %c0_119 = arith.constant 0 : index
      %c0_120 = arith.constant 0 : index
      %view_121 = memref.view %alloca_117[%c0_120][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_118, %view_121[%c0_119] : memref<1xindex>
      %c8_122 = arith.constant 8 : index
      %c1_i64_123 = arith.constant 1 : i64
      %c8_124 = arith.constant 8 : index
      %c1_i64_125 = arith.constant 1 : i64
      %c1_126 = arith.constant 1 : index
      %13 = arith.muli %c8_124, %c1_126 : index
      %subview_127 = memref.subview %alloca_117[%13] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_128 = arith.constant 0 : index
      %base_buffer_129, %offset_130, %sizes_131, %strides_132 = memref.extract_strided_metadata %subview_127 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_133 = memref.reinterpret_cast %base_buffer_129 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_134 = memref.view %reinterpret_cast_133[%offset_130][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg12, %view_134[%c0_128] : memref<1xi64>
      %c0_i8_135 = arith.constant 0 : i8
      %c0_136 = arith.constant 0 : index
      memref.store %c0_i8_135, %alloca[%c0_136] : memref<1xi8>
      %c0_137 = arith.constant 0 : index
      %14 = memref.load %alloca[%c0_137] : memref<1xi8>
      %15 = func.call @_ZN3gpu3dim9block_dim17h409bcc0cefa30d66E(%14) : (i8) -> i64
      %c0_i8_138 = arith.constant 0 : i8
      %c0_139 = arith.constant 0 : index
      memref.store %c0_i8_138, %alloca_39[%c0_139] : memref<1xi8>
      %c0_140 = arith.constant 0 : index
      %16 = memref.load %alloca_39[%c0_140] : memref<1xi8>
      %17 = func.call @_ZN3gpu3dim8block_id17hd40c964ff5c88ab7E(%16) : (i8) -> i64
      %18 = arith.muli %15, %17 : i64
      %false = arith.constant false
      %c0_i8_141 = arith.constant 0 : i8
      %c0_142 = arith.constant 0 : index
      memref.store %c0_i8_141, %alloca_40[%c0_142] : memref<1xi8>
      %c0_143 = arith.constant 0 : index
      %19 = memref.load %alloca_40[%c0_143] : memref<1xi8>
      %20 = func.call @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E(%19) : (i8) -> i64
      %21 = arith.addi %18, %20 : i64
      %false_144 = arith.constant false
      %alloca_145 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_146 = arith.constant 0 : index
      %c0_147 = arith.constant 0 : index
      %view_148 = memref.view %alloca_145[%c0_147][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %21, %view_148[%c0_146] : memref<1xi64>
      %22 = arith.muli %21, %arg10 : i64
      %false_149 = arith.constant false
      %23 = arith.addi %arg10, %22 : i64
      %c1_i64_150 = arith.constant 1 : i64
      %24 = arith.subi %23, %c1_i64_150 : i64
      %25 = arith.subi %24, %arg9 : i64
      %c63_i64 = arith.constant 63 : i64
      %26 = arith.shrsi %25, %c63_i64 : i64
      %intptr_151 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %27 = arith.index_cast %intptr_151 : index to i64
      %28 = arith.andi %27, %26 : i64
      %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
      %30 = llvm.load volatile %29 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4 = arith.constant 4 : index
      %c4_152 = arith.constant 4 : index
      %c4_i64 = arith.constant 4 : i64
      %31 = arith.muli %22, %c4_i64 : i64
      %c4_i64_153 = arith.constant 4 : i64
      %c4_i64_154 = arith.constant 4 : i64
      %32 = arith.muli %22, %c4_i64_154 : i64
      %33 = arith.index_cast %32 : i64 to index
      %subview_155 = memref.subview %arg8[%33] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %alloca_156 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_157 = memref.extract_aligned_pointer_as_index %subview_155 : memref<4xi8, strided<[1], offset: ?>> -> index
      %c0_158 = arith.constant 0 : index
      %c0_159 = arith.constant 0 : index
      %view_160 = memref.view %alloca_156[%c0_159][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_157, %view_160[%c0_158] : memref<1xindex>
      %c8_161 = arith.constant 8 : index
      %c1_i64_162 = arith.constant 1 : i64
      %c8_163 = arith.constant 8 : index
      %c1_i64_164 = arith.constant 1 : i64
      %c1_165 = arith.constant 1 : index
      %34 = arith.muli %c8_163, %c1_165 : index
      %subview_166 = memref.subview %alloca_156[%34] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_167 = arith.constant 0 : index
      %base_buffer_168, %offset_169, %sizes_170, %strides_171 = memref.extract_strided_metadata %subview_166 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_172 = memref.reinterpret_cast %base_buffer_168 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_173 = memref.view %reinterpret_cast_172[%offset_169][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg10, %view_173[%c0_167] : memref<1xi64>
      %35 = arith.muli %21, %arg5 : i64
      %false_174 = arith.constant false
      %36 = arith.addi %arg5, %35 : i64
      %c1_i64_175 = arith.constant 1 : i64
      %37 = arith.subi %36, %c1_i64_175 : i64
      %38 = arith.subi %37, %arg4 : i64
      %c63_i64_176 = arith.constant 63 : i64
      %39 = arith.shrsi %38, %c63_i64_176 : i64
      %intptr_177 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %40 = arith.index_cast %intptr_177 : index to i64
      %41 = arith.andi %40, %39 : i64
      %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
      %43 = llvm.load volatile %42 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4_178 = arith.constant 4 : index
      %c4_179 = arith.constant 4 : index
      %c4_i64_180 = arith.constant 4 : i64
      %44 = arith.muli %35, %c4_i64_180 : i64
      %c4_i64_181 = arith.constant 4 : i64
      %c4_i64_182 = arith.constant 4 : i64
      %45 = arith.muli %35, %c4_i64_182 : i64
      %46 = arith.index_cast %45 : i64 to index
      %subview_183 = memref.subview %arg3[%46] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %alloca_184 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_185 = memref.extract_aligned_pointer_as_index %subview_183 : memref<4xi8, strided<[1], offset: ?>> -> index
      %c0_186 = arith.constant 0 : index
      %c0_187 = arith.constant 0 : index
      %view_188 = memref.view %alloca_184[%c0_187][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_185, %view_188[%c0_186] : memref<1xindex>
      %c8_189 = arith.constant 8 : index
      %c1_i64_190 = arith.constant 1 : i64
      %c8_191 = arith.constant 8 : index
      %c1_i64_192 = arith.constant 1 : i64
      %c1_193 = arith.constant 1 : index
      %47 = arith.muli %c8_191, %c1_193 : index
      %subview_194 = memref.subview %alloca_184[%47] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_195 = arith.constant 0 : index
      %base_buffer_196, %offset_197, %sizes_198, %strides_199 = memref.extract_strided_metadata %subview_194 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_200 = memref.reinterpret_cast %base_buffer_196 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_201 = memref.view %reinterpret_cast_200[%offset_197][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg5, %view_201[%c0_195] : memref<1xi64>
      %48 = arith.muli %21, %arg2 : i64
      %false_202 = arith.constant false
      %49 = arith.addi %arg2, %48 : i64
      %c1_i64_203 = arith.constant 1 : i64
      %50 = arith.subi %49, %c1_i64_203 : i64
      %51 = arith.subi %50, %arg1 : i64
      %c63_i64_204 = arith.constant 63 : i64
      %52 = arith.shrsi %51, %c63_i64_204 : i64
      %intptr_205 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %53 = arith.index_cast %intptr_205 : index to i64
      %54 = arith.andi %53, %52 : i64
      %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
      %56 = llvm.load volatile %55 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4_206 = arith.constant 4 : index
      %c4_207 = arith.constant 4 : index
      %c4_i64_208 = arith.constant 4 : i64
      %57 = arith.muli %48, %c4_i64_208 : i64
      %c4_i64_209 = arith.constant 4 : i64
      %c4_i64_210 = arith.constant 4 : i64
      %58 = arith.muli %48, %c4_i64_210 : i64
      %59 = arith.index_cast %58 : i64 to index
      %subview_211 = memref.subview %arg0[%59] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %alloca_212 = memref.alloca() {alignment = 8 : i64} : memref<16xi8>
      %intptr_213 = memref.extract_aligned_pointer_as_index %subview_211 : memref<4xi8, strided<[1], offset: ?>> -> index
      %c0_214 = arith.constant 0 : index
      %c0_215 = arith.constant 0 : index
      %view_216 = memref.view %alloca_212[%c0_215][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<16xi8> to memref<1xindex>
      memref.store %intptr_213, %view_216[%c0_214] : memref<1xindex>
      %c8_217 = arith.constant 8 : index
      %c1_i64_218 = arith.constant 1 : i64
      %c8_219 = arith.constant 8 : index
      %c1_i64_220 = arith.constant 1 : i64
      %c1_221 = arith.constant 1 : index
      %60 = arith.muli %c8_219, %c1_221 : index
      %subview_222 = memref.subview %alloca_212[%60] [1] [1] : memref<16xi8> to memref<1xi8, strided<[1], offset: ?>>
      %c0_223 = arith.constant 0 : index
      %base_buffer_224, %offset_225, %sizes_226, %strides_227 = memref.extract_strided_metadata %subview_222 : memref<1xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_228 = memref.reinterpret_cast %base_buffer_224 to offset: [0], sizes: [1], strides: [1] : memref<i8> to memref<1xi8, strided<[1]>>
      %view_229 = memref.view %reinterpret_cast_228[%offset_225][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<1xi8, strided<[1]>> to memref<1xi64>
      memref.store %arg2, %view_229[%c0_223] : memref<1xi64>
      %c0_i8_230 = arith.constant 0 : i8
      %c0_231 = arith.constant 0 : index
      memref.store %c0_i8_230, %alloca_41[%c0_231] : memref<1xi8>
      %c0_232 = arith.constant 0 : index
      %61 = memref.load %alloca_41[%c0_232] : memref<1xi8>
      %62 = func.call @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E(%61) : (i8) -> i64
      %alloca_233 = memref.alloca() {alignment = 8 : i64} : memref<8xi8>
      %c0_234 = arith.constant 0 : index
      %c0_235 = arith.constant 0 : index
      %view_236 = memref.view %alloca_233[%c0_235][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<8xi8> to memref<1xi64>
      memref.store %62, %view_236[%c0_234] : memref<1xi64>
      %c0_i64_237 = arith.constant 0 : i64
      %c0_i64_238 = arith.constant 0 : i64
      %63 = arith.cmpi ult, %c0_i64_238, %arg2 : i64
      %64 = arith.subi %c0_i64_237, %arg2 : i64
      %c63_i64_239 = arith.constant 63 : i64
      %65 = arith.shrsi %64, %c63_i64_239 : i64
      %intptr_240 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %66 = arith.index_cast %intptr_240 : index to i64
      %67 = arith.andi %66, %65 : i64
      %68 = llvm.inttoptr %67 : i64 to !llvm.ptr
      %69 = llvm.load volatile %68 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_241 = arith.constant 0 : i64
      %c4_i64_242 = arith.constant 4 : i64
      %c0_i64_243 = arith.constant 0 : i64
      %c4_i64_244 = arith.constant 4 : i64
      %70 = arith.muli %c0_i64_243, %c4_i64_244 : i64
      %71 = arith.index_cast %70 : i64 to index
      %base_buffer_245, %offset_246, %sizes_247, %strides_248 = memref.extract_strided_metadata %subview_211 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_249 = memref.reinterpret_cast %base_buffer_245 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_250 = memref.view %reinterpret_cast_249[%offset_246][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_251 = memref.subview %view_250[%71] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %base_buffer_252, %offset_253, %sizes_254, %strides_255 = memref.extract_strided_metadata %subview_251 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_256 = memref.reinterpret_cast %base_buffer_252 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_257 = memref.view %reinterpret_cast_256[%offset_253][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %c0_258 = arith.constant 0 : index
      %72 = memref.load %view_257[%c0_258] : memref<1xi32>
      %73 = arith.cmpi ult, %62, %arg7 : i64
      %74 = arith.subi %62, %arg7 : i64
      %c63_i64_259 = arith.constant 63 : i64
      %75 = arith.shrsi %74, %c63_i64_259 : i64
      %intptr_260 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %76 = arith.index_cast %intptr_260 : index to i64
      %77 = arith.andi %76, %75 : i64
      %78 = llvm.inttoptr %77 : i64 to !llvm.ptr
      %79 = llvm.load volatile %78 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4_i64_261 = arith.constant 4 : i64
      %c4_i64_262 = arith.constant 4 : i64
      %80 = arith.muli %62, %c4_i64_262 : i64
      %81 = arith.index_cast %80 : i64 to index
      %subview_263 = memref.subview %arg6[%81] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %base_buffer_264, %offset_265, %sizes_266, %strides_267 = memref.extract_strided_metadata %subview_263 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_268 = memref.reinterpret_cast %base_buffer_264 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_269 = memref.view %reinterpret_cast_268[%offset_265][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %c0_270 = arith.constant 0 : index
      %82 = memref.load %view_269[%c0_270] : memref<1xi32>
      %83 = arith.addi %72, %82 : i32
      %false_271 = arith.constant false
      %c0_i64_272 = arith.constant 0 : i64
      %c0_i64_273 = arith.constant 0 : i64
      %84 = arith.cmpi ult, %c0_i64_273, %arg5 : i64
      %85 = arith.subi %c0_i64_272, %arg5 : i64
      %c63_i64_274 = arith.constant 63 : i64
      %86 = arith.shrsi %85, %c63_i64_274 : i64
      %intptr_275 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %87 = arith.index_cast %intptr_275 : index to i64
      %88 = arith.andi %87, %86 : i64
      %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
      %90 = llvm.load volatile %89 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_276 = arith.constant 0 : i64
      %c4_i64_277 = arith.constant 4 : i64
      %c0_i64_278 = arith.constant 0 : i64
      %c4_i64_279 = arith.constant 4 : i64
      %91 = arith.muli %c0_i64_278, %c4_i64_279 : i64
      %92 = arith.index_cast %91 : i64 to index
      %base_buffer_280, %offset_281, %sizes_282, %strides_283 = memref.extract_strided_metadata %subview_183 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_284 = memref.reinterpret_cast %base_buffer_280 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_285 = memref.view %reinterpret_cast_284[%offset_281][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_286 = memref.subview %view_285[%92] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %c0_287 = arith.constant 0 : index
      %base_buffer_288, %offset_289, %sizes_290, %strides_291 = memref.extract_strided_metadata %subview_286 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_292 = memref.reinterpret_cast %base_buffer_288 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_293 = memref.view %reinterpret_cast_292[%offset_289][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      memref.store %83, %view_293[%c0_287] : memref<1xi32>
      %c0_i64_294 = arith.constant 0 : i64
      %c0_i64_295 = arith.constant 0 : i64
      %93 = arith.cmpi ult, %c0_i64_295, %arg5 : i64
      %94 = arith.subi %c0_i64_294, %arg5 : i64
      %c63_i64_296 = arith.constant 63 : i64
      %95 = arith.shrsi %94, %c63_i64_296 : i64
      %intptr_297 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %96 = arith.index_cast %intptr_297 : index to i64
      %97 = arith.andi %96, %95 : i64
      %98 = llvm.inttoptr %97 : i64 to !llvm.ptr
      %99 = llvm.load volatile %98 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_298 = arith.constant 0 : i64
      %c4_i64_299 = arith.constant 4 : i64
      %c0_i64_300 = arith.constant 0 : i64
      %c4_i64_301 = arith.constant 4 : i64
      %100 = arith.muli %c0_i64_300, %c4_i64_301 : i64
      %101 = arith.index_cast %100 : i64 to index
      %base_buffer_302, %offset_303, %sizes_304, %strides_305 = memref.extract_strided_metadata %subview_183 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_306 = memref.reinterpret_cast %base_buffer_302 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_307 = memref.view %reinterpret_cast_306[%offset_303][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_308 = memref.subview %view_307[%101] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_309 = arith.constant 1 : i32
      %c0_310 = arith.constant 0 : index
      %base_buffer_311, %offset_312, %sizes_313, %strides_314 = memref.extract_strided_metadata %subview_308 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_315 = memref.reinterpret_cast %base_buffer_311 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_316 = memref.view %reinterpret_cast_315[%offset_312][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %102 = memref.atomic_rmw addi %c1_i32_309, %view_316[%c0_310] : (i32, memref<1xi32>) -> i32
      %c0_i64_317 = arith.constant 0 : i64
      %c0_i64_318 = arith.constant 0 : i64
      %103 = arith.cmpi ult, %c0_i64_318, %arg5 : i64
      %104 = arith.subi %c0_i64_317, %arg5 : i64
      %c63_i64_319 = arith.constant 63 : i64
      %105 = arith.shrsi %104, %c63_i64_319 : i64
      %intptr_320 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %106 = arith.index_cast %intptr_320 : index to i64
      %107 = arith.andi %106, %105 : i64
      %108 = llvm.inttoptr %107 : i64 to !llvm.ptr
      %109 = llvm.load volatile %108 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_321 = arith.constant 0 : i64
      %c4_i64_322 = arith.constant 4 : i64
      %c0_i64_323 = arith.constant 0 : i64
      %c4_i64_324 = arith.constant 4 : i64
      %110 = arith.muli %c0_i64_323, %c4_i64_324 : i64
      %111 = arith.index_cast %110 : i64 to index
      %base_buffer_325, %offset_326, %sizes_327, %strides_328 = memref.extract_strided_metadata %subview_183 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_329 = memref.reinterpret_cast %base_buffer_325 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_330 = memref.view %reinterpret_cast_329[%offset_326][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_331 = memref.subview %view_330[%111] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %base_buffer_332, %offset_333, %sizes_334, %strides_335 = memref.extract_strided_metadata %subview_331 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_336 = memref.reinterpret_cast %base_buffer_332 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_337 = memref.view %reinterpret_cast_336[%offset_333][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      %c0_338 = arith.constant 0 : index
      %112 = memref.load %view_337[%c0_338] : memref<1xi32>
      %c30_i32 = arith.constant 30 : i32
      %c30_i32_339 = arith.constant 30 : i32
      %113 = arith.addi %112, %c30_i32_339 : i32
      %false_340 = arith.constant false
      %alloca_341 = memref.alloca() {alignment = 4 : i64} : memref<4xi8>
      %c0_342 = arith.constant 0 : index
      %c0_343 = arith.constant 0 : index
      %view_344 = memref.view %alloca_341[%c0_343][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xi32>
      memref.store %113, %view_344[%c0_342] : memref<1xi32>
      %114 = llvm.inline_asm has_side_effects is_align_stack "mov.u32 ${0}, ${1};", "=&r,r,~{memory}" %113 : (i32) -> i32
      %c0_345 = arith.constant 0 : index
      %c0_346 = arith.constant 0 : index
      %view_347 = memref.view %alloca_42[%c0_346][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xi32>
      memref.store %114, %view_347[%c0_345] : memref<1xi32>
      %c0_348 = arith.constant 0 : index
      %view_349 = memref.view %alloca_42[%c0_348][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8> to memref<1xi32>
      %c0_350 = arith.constant 0 : index
      %115 = memref.load %view_349[%c0_350] : memref<1xi32>
      %c0_i64_351 = arith.constant 0 : i64
      %c0_i64_352 = arith.constant 0 : i64
      %116 = arith.cmpi ult, %c0_i64_352, %arg5 : i64
      %117 = arith.subi %c0_i64_351, %arg5 : i64
      %c63_i64_353 = arith.constant 63 : i64
      %118 = arith.shrsi %117, %c63_i64_353 : i64
      %intptr_354 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %119 = arith.index_cast %intptr_354 : index to i64
      %120 = arith.andi %119, %118 : i64
      %121 = llvm.inttoptr %120 : i64 to !llvm.ptr
      %122 = llvm.load volatile %121 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_355 = arith.constant 0 : i64
      %c4_i64_356 = arith.constant 4 : i64
      %c0_i64_357 = arith.constant 0 : i64
      %c4_i64_358 = arith.constant 4 : i64
      %123 = arith.muli %c0_i64_357, %c4_i64_358 : i64
      %124 = arith.index_cast %123 : i64 to index
      %base_buffer_359, %offset_360, %sizes_361, %strides_362 = memref.extract_strided_metadata %subview_183 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_363 = memref.reinterpret_cast %base_buffer_359 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_364 = memref.view %reinterpret_cast_363[%offset_360][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_365 = memref.subview %view_364[%124] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %c0_366 = arith.constant 0 : index
      %base_buffer_367, %offset_368, %sizes_369, %strides_370 = memref.extract_strided_metadata %subview_365 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_371 = memref.reinterpret_cast %base_buffer_367 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_372 = memref.view %reinterpret_cast_371[%offset_368][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xi32>
      memref.store %115, %view_372[%c0_366] : memref<1xi32>
      %125 = arith.cmpi ult, %62, %arg12 : i64
      %126 = arith.subi %62, %arg12 : i64
      %c63_i64_373 = arith.constant 63 : i64
      %127 = arith.shrsi %126, %c63_i64_373 : i64
      %intptr_374 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %128 = arith.index_cast %intptr_374 : index to i64
      %129 = arith.andi %128, %127 : i64
      %130 = llvm.inttoptr %129 : i64 to !llvm.ptr
      %131 = llvm.load volatile %130 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c4_i64_375 = arith.constant 4 : i64
      %c4_i64_376 = arith.constant 4 : i64
      %132 = arith.muli %62, %c4_i64_376 : i64
      %133 = arith.index_cast %132 : i64 to index
      %subview_377 = memref.subview %arg11[%133] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %base_buffer_378, %offset_379, %sizes_380, %strides_381 = memref.extract_strided_metadata %subview_377 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_382 = memref.reinterpret_cast %base_buffer_378 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_383 = memref.view %reinterpret_cast_382[%offset_379][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %134 = func.call @_ZN3gpu10__ldcs_f3217h82bc33ad1a5bfb48E(%view_383) : (memref<4xi8>) -> f32
      %c0_i64_384 = arith.constant 0 : i64
      %c0_i64_385 = arith.constant 0 : i64
      %135 = arith.cmpi ult, %c0_i64_385, %arg10 : i64
      %136 = arith.subi %c0_i64_384, %arg10 : i64
      %c63_i64_386 = arith.constant 63 : i64
      %137 = arith.shrsi %136, %c63_i64_386 : i64
      %intptr_387 = memref.extract_aligned_pointer_as_index %alloca_43 : memref<8xi8> -> index
      %138 = arith.index_cast %intptr_387 : index to i64
      %139 = arith.andi %138, %137 : i64
      %140 = llvm.inttoptr %139 : i64 to !llvm.ptr
      %141 = llvm.load volatile %140 {alignment = 8 : i64} : !llvm.ptr -> i8
      %c0_i64_388 = arith.constant 0 : i64
      %c4_i64_389 = arith.constant 4 : i64
      %c0_i64_390 = arith.constant 0 : i64
      %c4_i64_391 = arith.constant 4 : i64
      %142 = arith.muli %c0_i64_390, %c4_i64_391 : i64
      %143 = arith.index_cast %142 : i64 to index
      %base_buffer_392, %offset_393, %sizes_394, %strides_395 = memref.extract_strided_metadata %subview_155 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_396 = memref.reinterpret_cast %base_buffer_392 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_397 = memref.view %reinterpret_cast_396[%offset_393][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<4xi8>
      %subview_398 = memref.subview %view_397[%143] [4] [1] : memref<4xi8> to memref<4xi8, strided<[1], offset: ?>>
      %c0_399 = arith.constant 0 : index
      %base_buffer_400, %offset_401, %sizes_402, %strides_403 = memref.extract_strided_metadata %subview_398 : memref<4xi8, strided<[1], offset: ?>> -> memref<i8>, index, index, index
      %reinterpret_cast_404 = memref.reinterpret_cast %base_buffer_400 to offset: [0], sizes: [4], strides: [1] : memref<i8> to memref<4xi8, strided<[1]>>
      %view_405 = memref.view %reinterpret_cast_404[%offset_401][] {operand_segment_sizes = array<i32: 1, 1, 0>} : memref<4xi8, strided<[1]>> to memref<1xf32>
      memref.store %134, %view_405[%c0_399] : memref<1xf32>
      gpu.return
    }
    memref.global constant @memory_alloc_1 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %0 = memref.get_global @memory_alloc_1 : memref<11xi8>
    func.func private @_ZN3gpu20add_mlir_string_attr17h0b16585ac8dd7908E(memref<1xi8>, i64) -> i64 attributes {gpu_codegen_builtin = "gpu::add_mlir_string_attr"}
    %f_31 = func.constant {gpu_codegen_builtin = "gpu::add_mlir_string_attr"} @_ZN3gpu20add_mlir_string_attr17h0b16585ac8dd7908E : (memref<1xi8>, i64) -> i64
    func.func private @_ZN3gpu3dim9_block_id17h9a3bf55096aa3164E() -> i64 attributes {gpu_codegen_builtin = "gpu::block_id"}
    %f_32 = func.constant {gpu_codegen_builtin = "gpu::block_id"} @_ZN3gpu3dim9_block_id17h9a3bf55096aa3164E : () -> i64
    memref.global constant @memory_alloc_2 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 121, 62]>
    %1 = memref.get_global @memory_alloc_2 : memref<11xi8>
    memref.global constant @memory_alloc_3 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 122, 62]>
    %2 = memref.get_global @memory_alloc_3 : memref<11xi8>
    memref.global constant @memory_alloc_4 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %3 = memref.get_global @memory_alloc_4 : memref<11xi8>
    func.func private @_ZN3gpu3dim10_block_dim17h8a4ce5d49d624b14E() -> i64 attributes {gpu_codegen_builtin = "gpu::block_dim"}
    %f_33 = func.constant {gpu_codegen_builtin = "gpu::block_dim"} @_ZN3gpu3dim10_block_dim17h8a4ce5d49d624b14E : () -> i64
    memref.global constant @memory_alloc_5 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 121, 62]>
    %4 = memref.get_global @memory_alloc_5 : memref<11xi8>
    memref.global constant @memory_alloc_6 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 122, 62]>
    %5 = memref.get_global @memory_alloc_6 : memref<11xi8>
    memref.global constant @memory_alloc_7 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 120, 62]>
    %6 = memref.get_global @memory_alloc_7 : memref<11xi8>
    func.func private @_ZN3gpu3dim10_thread_id17h1829fde4294be6f3E() -> i64 attributes {gpu_codegen_builtin = "gpu::thread_id"}
    %f_34 = func.constant {gpu_codegen_builtin = "gpu::thread_id"} @_ZN3gpu3dim10_thread_id17h1829fde4294be6f3E : () -> i64
    memref.global constant @memory_alloc_8 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 121, 62]>
    %7 = memref.get_global @memory_alloc_8 : memref<11xi8>
    memref.global constant @memory_alloc_9 : memref<11xi8> = dense<[35, 103, 112, 117, 60, 100, 105, 109, 32, 122, 62]>
    %8 = memref.get_global @memory_alloc_9 : memref<11xi8>
    %f_35 = func.constant @_ZN3gpu3dim9block_dim17h409bcc0cefa30d66E : (i8) -> i64
    %f_36 = func.constant @_ZN3gpu3dim8block_id17hd40c964ff5c88ab7E : (i8) -> i64
    %f_37 = func.constant @_ZN3gpu3dim9thread_id17hd50eec0b02fa7676E : (i8) -> i64
    %f_38 = func.constant @_ZN3gpu10__ldcs_f3217h82bc33ad1a5bfb48E : (memref<4xi8>) -> f32
  }
  module {
  }
  func.func private @_ZN3gpu10atomic_add17h3a9c21a4f17e3e4bE(memref<4xi8>, i32) -> i32 attributes {gpu_codegen_builtin = "gpu::atomic_add"}
  func.func private @_ZN3gpu12subslice_mut17h1242923c386eb999E(memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64) attributes {gpu_codegen_builtin = "gpu::subslice_mut"}
  func.func private @_ZN3gpu12subslice_mut17h8501f848cdb8fb24E(memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64) attributes {gpu_codegen_builtin = "gpu::subslice_mut"}
  func.func private @_ZN3gpu8subslice17h1cbe85c5a0ff6edfE(memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64) attributes {gpu_codegen_builtin = "gpu::subslice"}
  %c0_i64 = arith.constant 0 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c11 = arith.constant 11 : index
  %c11_2 = arith.constant 11 : index
  %c11_3 = arith.constant 11 : index
  %c11_4 = arith.constant 11 : index
  %c11_5 = arith.constant 11 : index
  %c11_6 = arith.constant 11 : index
  %c11_7 = arith.constant 11 : index
  %c11_8 = arith.constant 11 : index
  %c11_9 = arith.constant 11 : index
  %c0_i64_10 = arith.constant 0 : i64
  %c8 = arith.constant 8 : index
  %c0_i64_11 = arith.constant 0 : i64
  %c8_12 = arith.constant 8 : index
  %c0_i64_13 = arith.constant 0 : i64
  %c8_14 = arith.constant 8 : index
  %c0_i64_15 = arith.constant 0 : i64
  %c8_16 = arith.constant 8 : index
  %c0_i64_17 = arith.constant 0 : i64
  %c8_18 = arith.constant 8 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i8_19 = arith.constant 0 : i8
  %c0_i8_20 = arith.constant 0 : i8
  %f = func.constant {gpu_codegen_builtin = "gpu::subslice_mut"} @_ZN3gpu12subslice_mut17h1242923c386eb999E : (memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64)
  %c0_i64_21 = arith.constant 0 : i64
  %c8_22 = arith.constant 8 : index
  %f_23 = func.constant {gpu_codegen_builtin = "gpu::subslice_mut"} @_ZN3gpu12subslice_mut17h8501f848cdb8fb24E : (memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64)
  %c0_i64_24 = arith.constant 0 : i64
  %c8_25 = arith.constant 8 : index
  %f_26 = func.constant {gpu_codegen_builtin = "gpu::subslice"} @_ZN3gpu8subslice17h1cbe85c5a0ff6edfE : (memref<4xi8>, i64, i64, i64) -> (memref<4xi8>, i64)
  %c0_i64_27 = arith.constant 0 : i64
  %c8_28 = arith.constant 8 : index
  %c0_i8_29 = arith.constant 0 : i8
  %f_30 = func.constant {gpu_codegen_builtin = "gpu::atomic_add"} @_ZN3gpu10atomic_add17h3a9c21a4f17e3e4bE : (memref<4xi8>, i32) -> i32
}
