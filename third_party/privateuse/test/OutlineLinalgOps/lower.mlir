// RUN: privateuse-opt %s -privateuse-outline-linalg-ops | FileCheck %s
// RUN: privateuse-opt %s -privateuse-outline-linalg-ops="outlined-attr-value=OUTLINED" \
// RUN:   | FileCheck %s --check-prefix=ATTR
// RUN: privateuse-opt %s -privateuse-outline-linalg-ops -privateuse-lower-linalg-reduce \
// RUN:   -convert-linalg-to-loops -privateuse-memref-to-pumem \
// RUN:   | FileCheck %s --check-prefix=PUMEM --implicit-check-not=memref.

// Every top-level buffer linalg op moves into a private function named after
// the op and its body ops; the call site passes the operands and any values
// the body captured from the enclosing scope. Outlining makes the buffers
// explicit in the signature so later passes can lower whole ops at a time.

#identity = affine_map<(d0) -> (d0)>

// The outlined functions are inserted at the top of the module in reverse
// outlining order: the fill is outlined first, the generic second.

// CHECK-LABEL: func.func private @outlined_linalg_generic_addf_1(
// CHECK-SAME:  %[[IN:.*]]: memref<16xf32>, %[[OUT:.*]]: memref<16xf32>
// CHECK:       linalg.generic
// CHECK:         arith.addf
// CHECK:       return
//
// CHECK-LABEL: func.func private @outlined_linalg_fill_0(
// CHECK-SAME:  %[[VALUE:.*]]: f32, %[[OUT:.*]]: memref<16xf32>
// CHECK:       linalg.fill ins(%[[VALUE]] : f32) outs(%[[OUT]] : memref<16xf32>)
// CHECK:       return

// With a marker value the outlined functions carry it, while the kernel that
// stays behind does not.

// ATTR-LABEL: func.func private @outlined_linalg_generic_addf_1(
// ATTR-SAME:  %{{.*}}: memref<16xf32>, %{{.*}}: memref<16xf32>) attributes {pumem.outlined = "OUTLINED"}
// ATTR-LABEL: func.func private @outlined_linalg_fill_0(
// ATTR-SAME:  %{{.*}}: f32, %{{.*}}: memref<16xf32>) attributes {pumem.outlined = "OUTLINED"}
// ATTR-LABEL: func.func @kernel(
// ATTR-NOT:   pumem.outlined

func.func @kernel(%a: memref<16xf32>, %b: memref<16xf32>) {
  // CHECK-LABEL: func.func @kernel(
  // CHECK-NOT:   linalg.
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%b : memref<16xf32>)
  linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"]} ins(%a : memref<16xf32>) outs(%b : memref<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  }
  // CHECK:       call @outlined_linalg_fill_0({{.*}}) : (f32, memref<16xf32>) -> ()
  // CHECK:       call @outlined_linalg_generic_addf_1({{.*}}) : (memref<16xf32>, memref<16xf32>) -> ()
  // CHECK:       return
  return
}

// After the memref to pumem conversion the outlined signatures and the calls
// carry the scalar tuple ABI as well. The static memref sizes are folded into
// the body, so outlined functions only keep the descriptor components they
// still read.

// PUMEM-LABEL: func.func private @outlined_linalg_generic_addf_1(
// PUMEM-SAME:  %[[IN:[a-zA-Z0-9_]+]]: !pumem.ptr<f32, 0>
// PUMEM-SAME:  , %{{[a-zA-Z0-9_]+}}: index, %{{[a-zA-Z0-9_]+}}: index, %[[OUT:[a-zA-Z0-9_]+]]: !pumem.ptr<f32, 0>
// PUMEM-SAME:  , %{{[a-zA-Z0-9_]+}}: index, %{{[a-zA-Z0-9_]+}}: index) {
// PUMEM-NOT:   memref.
//
// PUMEM-LABEL: func.func private @outlined_linalg_fill_0(
// PUMEM-SAME:  %[[VALUE:[a-zA-Z0-9_]+]]: f32
// PUMEM-SAME:  , %[[BUF:[a-zA-Z0-9_]+]]: !pumem.ptr<f32, 0>, %{{[a-zA-Z0-9_]+}}: index, %{{[a-zA-Z0-9_]+}}: index) {
// PUMEM-NOT:   memref.
//
// PUMEM-LABEL: func.func @kernel(
// PUMEM:       call @outlined_linalg_fill_0({{.*}}) : (f32, !pumem.ptr<f32, 0>, index, index) -> ()
// PUMEM:       call @outlined_linalg_generic_addf_1({{.*}}) : (!pumem.ptr<f32, 0>, index, index, !pumem.ptr<f32, 0>, index, index) -> ()
