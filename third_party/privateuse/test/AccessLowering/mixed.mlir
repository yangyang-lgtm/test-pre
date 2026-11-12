// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-ptr-to-memref -canonicalize | FileCheck %s

module attributes {privateuse.grid = array<i64: 4, 3, 1>} {
  // 同一个函数里混有可分析与不可分析的访存：前者转 memref，后者保持 tt 形态。
  tt.func @mixed(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c4 = arith.constant 4 : i32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %c4S = tt.splat %c4 : i32 -> tensor<8xi32>
    %half = arith.cmpi slt, %r, %c4S : tensor<8xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %p = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %p : tensor<8x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %q = tt.addptr %qS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %q, %v : tensor<8x!tt.ptr<f32>>
    %m = tt.load %p, %half : tensor<8x!tt.ptr<f32>>
    tt.store %q, %m, %half : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  // 循环内的 load 步长是动态标量（不可分析），循环外的 store 可以分析。
  tt.func @dynamic_step(%p: !tt.ptr<f32>, %n: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %acc0 = tt.splat %cst : f32 -> tensor<128xf32>
    %stepS = tt.splat %step : i32 -> tensor<128xi32>
    %res:2 = scf.for %iv = %c0 to %n step %step iter_args(%ptr = %pS, %acc = %acc0) -> (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) : i32 {
      %v = tt.load %ptr : tensor<128x!tt.ptr<f32>>
      %sum = arith.addf %acc, %v : tensor<128xf32>
      %next = tt.addptr %ptr, %stepS : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      scf.yield %next, %sum : tensor<128x!tt.ptr<f32>>, tensor<128xf32>
    }
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %q = tt.addptr %pS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %q, %res#1 : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  // 循环携带的指针 iter_arg 只被未降级的 masked load 消费：init 的 +4 偏移
  // 必须保留（链不能被 token 化），循环保持 tensor-of-pointer 形态。
  tt.func @loop_carried(%p: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %c4S = tt.splat %c4 : i32 -> tensor<8xi32>
    %off = tt.addptr %pS, %c4S : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %acc0 = tt.splat %cst : f32 -> tensor<8xf32>
    %half = arith.cmpi slt, %r, %c4S : tensor<8xi32>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%ptr = %off, %acc = %acc0) -> (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) : i32 {
      %v = tt.load %ptr, %half : tensor<8x!tt.ptr<f32>>
      %sum = arith.addf %acc, %v : tensor<8xf32>
      %next = tt.addptr %ptr, %c4S : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      scf.yield %next, %sum : tensor<8x!tt.ptr<f32>>, tensor<8xf32>
    }
    %q = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %q, %res#1 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func @mixed(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK-DAG: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<?xf32> to !tt.ptr<f32>
// CHECK-DAG: %[[DST:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<?xf32> to !tt.ptr<f32>
// CHECK-DAG: %[[PS:.*]] = tt.splat %[[SRC]] : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-DAG: %[[P:.*]] = tt.addptr %[[PS]], %{{.*}} : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-DAG: %[[QS:.*]] = tt.splat %[[DST]] : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-DAG: %[[Q:.*]] = tt.addptr %[[QS]], %{{.*}} : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-DAG: memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
// CHECK-DAG: memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
// CHECK-DAG: memref.copy
// CHECK-DAG: memref.copy
// CHECK-DAG: tt.load %[[P]], %{{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK-DAG: tt.store %[[Q]], %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK: tt.return

// CHECK-LABEL: tt.func @dynamic_step(
// CHECK-SAME: %arg0: memref<?xf32>,
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<?xf32> to !tt.ptr<f32>
// CHECK: %[[P:.*]] = tt.splat %[[BASE]] : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK: scf.for {{.*}} iter_args(%[[PTR:.*]] = %[[P]], {{.*}} = {{.*}}) -> (tensor<128x!tt.ptr<f32>>, tensor<128xf32>)
// CHECK: tt.load %[[PTR]] : tensor<128x!tt.ptr<f32>>
// CHECK: tt.addptr %[[PTR]], {{.*}} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK: scf.yield
// CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1]
// CHECK: memref.copy
// CHECK: tt.return

// CHECK-LABEL: tt.func @loop_carried(
// CHECK-SAME: %arg0: memref<?xf32>,
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<?xf32> to !tt.ptr<f32>
// CHECK: %[[P:.*]] = tt.splat %[[BASE]] : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK: %[[OFF:.*]] = tt.addptr %[[P]], {{.*}} : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK: scf.for {{.*}} iter_args(%[[PTR:.*]] = %[[OFF]], {{.*}} = {{.*}}) -> (tensor<8x!tt.ptr<f32>>, tensor<8xf32>)
// CHECK: tt.load %[[PTR]], {{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK: tt.addptr %[[PTR]], {{.*}} : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK: scf.yield
// CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1]
// CHECK: memref.copy
// CHECK: tt.return
