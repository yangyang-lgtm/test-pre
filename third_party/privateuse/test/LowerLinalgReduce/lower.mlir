// RUN: privateuse-opt %s -privateuse-outline-linalg-ops -privateuse-lower-linalg-reduce \
// RUN:   | FileCheck %s

// A single-axis reduction generic becomes a nest of scf.for loops: one loop
// per parallel dimension and an inner loop over the reduced dimension. The
// accumulator starts from the first input element, so the loop runs from 1 to
// the extent and never reads the (uninitialized) init buffer; the final
// accumulator is stored back into the init buffer.

#identity = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#identity1 = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

// The outlined functions are inserted at the top of the module in reverse
// outlining order: @full_sum's reduce is outlined second.

// A full reduction has no parallel dimensions: only the reduction loop is
// emitted and the result is stored into the scalar init buffer.

// CHECK-LABEL: func.func private @outlined_linalg_generic_addf_1(
// CHECK-SAME:  %[[IN:.*]]: memref<16xf32>, %[[OUT:.*]]: memref<f32>
// CHECK-NOT:   scf.for
// CHECK:       %[[FIRST:.*]] = memref.load %[[IN]][%c0{{[_0-9]*}}] : memref<16xf32>
// CHECK:       scf.for %[[J:.*]] = %c1{{[_0-9]*}} to %c16{{[_0-9]*}} step %c1{{[_0-9]*}} iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (f32) {
// CHECK:         %[[V:.*]] = memref.load %[[IN]][%[[J]]] : memref<16xf32>
// CHECK:         %{{.*}} = arith.addf %[[V]], %[[ACC]] : f32
// CHECK:         scf.yield
// CHECK:       memref.store %{{.*}}, %[[OUT]][] : memref<f32>

// CHECK-LABEL: func.func private @outlined_linalg_generic_maxnumf_0(
// CHECK-SAME:  %[[IN:.*]]: memref<64x64xf32>, %[[OUT:.*]]: memref<64xf32>
// CHECK:       scf.for %[[I:.*]] = %c0{{[_0-9]*}} to %c64{{[_0-9]*}} step %c1{{[_0-9]*}} {
// CHECK:         %[[FIRST:.*]] = memref.load %[[IN]][%[[I]], %c0{{[_0-9]*}}] : memref<64x64xf32>
// CHECK:         scf.for %[[J:.*]] = %c1{{[_0-9]*}} to %c64{{[_0-9]*}} step %c1{{[_0-9]*}} iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (f32) {
// CHECK:           %[[V:.*]] = memref.load %[[IN]][%[[I]], %[[J]]] : memref<64x64xf32>
// CHECK:           %{{.*}} = arith.maxnumf %[[V]], %[[ACC]] : f32
// CHECK:           scf.yield
// CHECK:         memref.store %{{.*}}, %[[OUT]][%[[I]]] : memref<64xf32>

func.func @row_max(%in : memref<64x64xf32>, %out : memref<64xf32>) {
  linalg.generic {indexing_maps = [#identity, #row], iterator_types = ["parallel", "reduction"]} ins(%in : memref<64x64xf32>) outs(%out : memref<64xf32>) {
  ^bb0(%a: f32, %b: f32):
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  return
}

func.func @full_sum(%in : memref<16xf32>, %out : memref<f32>) {
  linalg.generic {indexing_maps = [#identity1, #scalar], iterator_types = ["reduction"]} ins(%in : memref<16xf32>) outs(%out : memref<f32>) {
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  }
  return
}
