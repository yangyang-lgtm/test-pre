// RUN: privateuse-opt %s -privateuse-linalg-to-runtime-call | FileCheck %s
// RUN: privateuse-opt %s -privateuse-outline-linalg-ops -privateuse-linalg-to-runtime-call \
// RUN:   -privateuse-lower-linalg-reduce -convert-linalg-to-loops -privateuse-memref-to-pumem \
// RUN:   | FileCheck %s --check-prefix=PUMEM --implicit-check-not=memref.

// linalg.matmul and linalg.transpose over statically shaped rank-2 memrefs
// become direct runtime calls: the buffers pass through as operands and the
// element type plus the extents become compile-time template arguments.

// CHECK-LABEL: func.func @matmul(
// CHECK:       pumem.runtime_call @runtime_matmul<[f32, 4 : index, 16 : index, 8 : index]> (%{{.*}}, %{{.*}}, %{{.*}}) : (memref<4x8xf32>, memref<8x16xf32>, memref<4x16xf32>) -> ()
func.func @matmul(%a: memref<4x8xf32>, %b: memref<8x16xf32>, %c: memref<4x16xf32>) {
  linalg.matmul ins(%a, %b : memref<4x8xf32>, memref<8x16xf32>) outs(%c : memref<4x16xf32>)
  return
}

// Only the identity-of-transpose [1, 0] is supported; the extents are read off
// the source shape.

// CHECK-LABEL: func.func @transpose(
// CHECK:       pumem.runtime_call @runtime_transpose<[f32, 4 : index, 8 : index]> (%{{.*}}, %{{.*}}) : (memref<4x8xf32>, memref<8x4xf32>) -> ()
func.func @transpose(%a: memref<4x8xf32>, %b: memref<8x4xf32>) {
  linalg.transpose ins(%a : memref<4x8xf32>) outs(%b : memref<8x4xf32>) permutation = [1, 0]
  return
}

// The outlined functions from -privateuse-outline-linalg-ops are inserted at
// the top of the module in reverse outlining order, and after
// -privateuse-memref-to-pumem every buffer operand is the offset-adjusted
// pointer from the tuple ABI.

// PUMEM-LABEL: func.func private @outlined_linalg_transpose_1(
// PUMEM:       pumem.runtime_call @runtime_transpose<[f32, 4 : index, 8 : index]> ({{.*}}) : (!pumem.ptr<f32, 0>, !pumem.ptr<f32, 0>) -> ()
//
// PUMEM-LABEL: func.func private @outlined_linalg_matmul_mulf_addf_0(
// PUMEM:       pumem.runtime_call @runtime_matmul<[f32, 4 : index, 16 : index, 8 : index]> ({{.*}}) : (!pumem.ptr<f32, 0>, !pumem.ptr<f32, 0>, !pumem.ptr<f32, 0>) -> ()
//
// PUMEM-LABEL: func.func @matmul(
// PUMEM:       call @outlined_linalg_matmul_mulf_addf_0(
//
// PUMEM-LABEL: func.func @transpose(
// PUMEM:       call @outlined_linalg_transpose_1(
