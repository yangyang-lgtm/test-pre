// RUN: privateuse-opt %s -expand-strided-metadata -privateuse-memref-to-pumem \
// RUN:   -canonicalize | FileCheck %s --implicit-check-not=memref.

// Shapes and layouts are normalized by -expand-strided-metadata first: subviews
// become extract_strided_metadata + reinterpret_cast, which this pass folds
// into the scalar tuple. A subview at [1, 2] with strides [2, 1] of a
// memref<8x16xf32, 128> points 18 elements into the underlying memory.

// CHECK-LABEL: func.func @subview(
// CHECK-SAME: %[[BASE:.*]]: !pumem.ptr<f32, 128>
tt.func @subview(%A: memref<8x16xf32, 128>) -> f32 {
  %s = memref.subview %A[1, 2] [4, 8] [2, 1] : memref<8x16xf32, 128> to memref<4x8xf32, strided<[32, 1], offset: 18>, 128>
  %c0 = arith.constant 0 : index
  // CHECK: %[[C18:.*]] = arith.constant 18 : index
  // CHECK: %[[PTR:.*]] = pumem.addptr %[[BASE]], %[[C18]] : !pumem.ptr<f32, 128>
  // CHECK: pumem.load %[[PTR]] : !pumem.ptr<f32, 128> -> f32
  %v = memref.load %s[%c0, %c0] : memref<4x8xf32, strided<[32, 1], offset: 18>, 128>
  tt.return %v : f32
}

// A rank-reduced subview derives its base pointer from the offset alone.

// CHECK-LABEL: func.func @rank_reduced(
tt.func @rank_reduced(%A: memref<8x16xf32, 128>) -> f32 {
  %s = memref.subview %A[3, 5] [1, 1] [1, 1] : memref<8x16xf32, 128> to memref<f32, strided<[], offset: 53>, 128>
  // CHECK: %[[V:.*]] = pumem.load
  // CHECK-SAME: : !pumem.ptr<f32, 128> -> f32
  %v = memref.load %s[] : memref<f32, strided<[], offset: 53>, 128>
  tt.return %v : f32
}

// extract_strided_metadata reports the source offset, sizes and strides; the
// base buffer is the underlying memory start (offset 0).

// CHECK-LABEL: func.func @metadata(
// CHECK-SAME: %[[PTR:.*]]: !pumem.ptr<f32, 128>, %[[OFF:.*]]: index, %[[SIZE:.*]]: index, %[[STRIDE:.*]]: index)
tt.func @metadata(%A: memref<?xf32, strided<[?], offset: ?>, 128>) -> (index, index, index) {
  %base, %off, %size, %stride = memref.extract_strided_metadata %A : memref<?xf32, strided<[?], offset: ?>, 128> -> memref<f32, 128>, index, index, index
  // CHECK: return %[[OFF]], %[[SIZE]], %[[STRIDE]] : index, index, index
  tt.return %off, %size, %stride : index, index, index
}
