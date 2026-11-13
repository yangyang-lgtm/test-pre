// RUN: privateuse-opt %s -privateuse-bufferize | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  // Bufferize-created allocation must use the SRAM memory space (128) and the
  // default alignment (64), matching the tiles the access lowering allocates.
  func.func @elementwise(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %0 = tensor.empty() : tensor<16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  // CHECK-LABEL: func.func @elementwise
  // CHECK:       %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<16xf32, 128>
  // CHECK:       linalg.generic {{.*}} outs(%[[ALLOC]] : memref<16xf32, {{.*}}128>)
  // CHECK-NOT:   tensor.empty()
}
