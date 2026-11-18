// RUN: privateuse-opt %s --privateuse-hoist-invariant-staging | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @hoist
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [64], strides: [1]
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<64xf32, 128>
// CHECK: memref.copy %[[VIEW]], %[[BUF]] : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 128>
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[BUF]] restrict writable : memref<64xf32, 128> to tensor<64xf32>
// CHECK: scf.for
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[T]] : tensor<64xf32>)
// CHECK: return
// CHECK-NOT: memref.alloc
func.func @hoist(%src: memref<?xf32, strided<[1], offset: ?>>, %dst: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %view = memref.reinterpret_cast %src to offset: [%c0], sizes: [64], strides: [1] : memref<?xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
  scf.for %i = %c0 to %c64 step %c1 {
    %buf = memref.alloc() : memref<64xf32, 128>
    memref.copy %view, %buf : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 128>
    %t = bufferization.to_tensor %buf restrict writable : memref<64xf32, 128> to tensor<64xf32>
    %out = bufferization.to_tensor %dst restrict writable : memref<64xf32> to tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%t : tensor<64xf32>) outs(%out : tensor<64xf32>) {
    ^bb0(%in: f32, %o: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
  }
  return
}

// CHECK-LABEL: func.func @dependent_source
// CHECK: scf.for
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<64xf32, 128>
// CHECK: memref.copy
func.func @dependent_source(%src: memref<?xf32, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %view = memref.reinterpret_cast %src to offset: [%i], sizes: [64], strides: [1] : memref<?xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
    %buf = memref.alloc() : memref<64xf32, 128>
    memref.copy %view, %buf : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 128>
  }
  return
}

// CHECK-LABEL: func.func @written_twice
// CHECK: scf.for
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<64xf32, 128>
// CHECK: memref.copy
// CHECK: memref.copy
func.func @written_twice(%src: memref<?xf32, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %view = memref.reinterpret_cast %src to offset: [%c0], sizes: [64], strides: [1] : memref<?xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
  scf.for %i = %c0 to %c64 step %c1 {
    %buf = memref.alloc() : memref<64xf32, 128>
    memref.copy %view, %buf : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 128>
    memref.copy %view, %buf : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 128>
  }
  return
}

// CHECK-LABEL: func.func @feeds_output
// CHECK: scf.for
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<64xf32, 128>
// CHECK: memref.copy
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.*}} : tensor<64xf32>)
func.func @feeds_output(%src: memref<?xf32, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %view = memref.reinterpret_cast %src to offset: [%c0], sizes: [64], strides: [1] : memref<?xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
  scf.for %i = %c0 to %c64 step %c1 {
    %buf = memref.alloc() : memref<64xf32, 128>
    memref.copy %view, %buf : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 128>
    %t = bufferization.to_tensor %buf restrict writable : memref<64xf32, 128> to tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%t : tensor<64xf32>) outs(%t : tensor<64xf32>) {
    ^bb0(%in: f32, %o: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
  }
  return
}

// CHECK-LABEL: func.func @aliasing_write
// CHECK: scf.for
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<64xf32, 128>
// CHECK: memref.copy
// CHECK: memref.copy
func.func @aliasing_write(%src: memref<?xf32, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %view = memref.reinterpret_cast %src to offset: [%c0], sizes: [64], strides: [1] : memref<?xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
  %other = memref.reinterpret_cast %src to offset: [%c1], sizes: [64], strides: [1] : memref<?xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
  scf.for %i = %c0 to %c64 step %c1 {
    %buf = memref.alloc() : memref<64xf32, 128>
    memref.copy %view, %buf : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 128>
    memref.copy %view, %other : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
  }
  return
}

// CHECK-LABEL: func.func @dynamic_shape
// CHECK: scf.for
// CHECK: %[[BUF:.*]] = memref.alloc(%{{.*}}) : memref<?xf32, 128>
// CHECK: memref.copy
func.func @dynamic_shape(%src: memref<?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %buf = memref.alloc(%n) : memref<?xf32, 128>
    memref.copy %src, %buf : memref<?xf32> to memref<?xf32, 128>
  }
  return
}
