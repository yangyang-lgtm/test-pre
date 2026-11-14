// RUN: privateuse-opt %s -privateuse-memref-to-pumem | FileCheck %s --implicit-check-not=memref.

// Function boundaries follow the same tuple ABI as every other memref use:
// arguments are expanded in place, a memref result becomes several results,
// and func.call / func.return are rewritten to match.

// CHECK-LABEL: func.func private @helper(
// CHECK-SAME:  %[[IN:.*]]: !pumem.ptr<f32, 0>, %[[OFF:.*]]: index, %[[SIZE:.*]]: index, %[[STRIDE:.*]]: index
// CHECK-SAME:  ) -> (!pumem.ptr<f32, 0>, index, index, index)
// CHECK:       %[[V:.*]] = pumem.load
// CHECK:       pumem.store
// CHECK:       return %[[IN]], %[[OFF]], %[[SIZE]], %[[STRIDE]] : !pumem.ptr<f32, 0>, index, index, index
func.func private @helper(%in : memref<16xf32>) -> memref<16xf32> {
  %c0 = arith.constant 0 : index
  %v = memref.load %in[%c0] : memref<16xf32>
  %w = arith.addf %v, %v : f32
  memref.store %w, %in[%c0] : memref<16xf32>
  return %in : memref<16xf32>
}

// CHECK-LABEL: func.func @caller(
// CHECK-SAME:  %[[BUF:.*]]: !pumem.ptr<f32, 0>, {{.*}}: index, {{.*}}: index, {{.*}}: index
// CHECK:       %[[R:.*]]:4 = call @helper(%[[BUF]], {{.*}}, {{.*}}, {{.*}}) : (!pumem.ptr<f32, 0>, index, index, index) -> (!pumem.ptr<f32, 0>, index, index, index)
// CHECK:       %[[V:.*]] = pumem.load
// CHECK:       pumem.store
func.func @caller(%buf : memref<16xf32>) {
  %r = call @helper(%buf) : (memref<16xf32>) -> memref<16xf32>
  %c0 = arith.constant 0 : index
  %v = memref.load %r[%c0] : memref<16xf32>
  memref.store %v, %buf[%c0] : memref<16xf32>
  return
}
