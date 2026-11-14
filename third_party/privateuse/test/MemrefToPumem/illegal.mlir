// RUN: not privateuse-opt %s -privateuse-memref-to-pumem 2>&1 | FileCheck %s

// The conversion is illegal-by-default: unsupported memref operations must be
// named instead of silently surviving.

tt.func @subview(%A: memref<8x16xf32, 128>) {
  // CHECK: failed to legalize operation 'memref.subview'
  %s = memref.subview %A[0, 0] [4, 8] [1, 1] : memref<8x16xf32, 128> to memref<4x8xf32, strided<[16, 1]>, 128>
  tt.return
}
