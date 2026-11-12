// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-ptr-to-memref | FileCheck %s \
// RUN:   --implicit-check-not=memref.

module attributes {privateuse.grid = array<i64: 4, 3, 1>} {
  // mask 把行、列两个维度耦合（rows >= cols），分析层拒绝，整函数保持 tt。
  tt.func @coupled_mask(%p: !tt.ptr<f32>) {
    %rows = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %rowsE = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %c8 = arith.constant 8 : i32
    %c8S = tt.splat %c8 : i32 -> tensor<8x1xi32>
    %rowOff = arith.muli %rowsE, %c8S : tensor<8x1xi32>
    %rowB = tt.broadcast %rowOff : tensor<8x1xi32> -> tensor<8x8xi32>
    %colsE = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %colB = tt.broadcast %colsE : tensor<1x8xi32> -> tensor<8x8xi32>
    %idx = arith.addi %rowB, %colB : tensor<8x8xi32>
    %mask = arith.cmpi sge, %rowB, %colB : tensor<8x8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %idx : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %v = tt.load %a, %mask : tensor<8x8x!tt.ptr<f32>>
    tt.store %a, %v, %mask : tensor<8x8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @atomic(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %a = tt.addptr %pS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %c = arith.constant 1.000000e+00 : f32
    %cS = tt.splat %c : f32 -> tensor<128xf32>
    %v = tt.atomic_rmw fadd, acq_rel, gpu, %a, %cS : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) -> tensor<128xf32>
    tt.return
  }

  tt.func @broadcast_store(%dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 5 : i32, end = 6 : i32} : tensor<1xi32>
    %bc = tt.broadcast %r : tensor<1xi32> -> tensor<8xi32>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %q = tt.addptr %qS, %bc : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %c = arith.constant 1.000000e+00 : f32
    %cS = tt.splat %c : f32 -> tensor<8xf32>
    tt.store %q, %cS : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @negative_stride(%p: !tt.ptr<f32>) {
    %c127 = arith.constant 127 : i32
    %c127S = tt.splat %c127 : i32 -> tensor<128xi32>
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %rev = arith.subi %c127S, %r : tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %a = tt.addptr %pS, %rev : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %a : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  // dims [2,2] 的 strides [1,1] 不是单射（i+j 重叠），且无 mask 可裁出单射窗口，store 不能降级。
  tt.func @overlap_store(%dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %rE = tt.expand_dims %r {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %rB = tt.broadcast %rE : tensor<2x1xi32> -> tensor<2x2xi32>
    %cE = tt.expand_dims %r {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %cB = tt.broadcast %cE : tensor<1x2xi32> -> tensor<2x2xi32>
    %idx = arith.addi %rB, %cB : tensor<2x2xi32>
    %dS = tt.splat %dst : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %q = tt.addptr %dS, %idx : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %c = arith.constant 1.000000e+00 : f32
    %cS = tt.splat %c : f32 -> tensor<2x2xf32>
    tt.store %q, %cS : tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func @coupled_mask(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK-DAG: tt.load
// CHECK-DAG: tt.store

// CHECK-LABEL: tt.func @atomic(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK: tt.atomic_rmw

// CHECK-LABEL: tt.func @broadcast_store(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK: tt.store

// CHECK-LABEL: tt.func @negative_stride(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK: tt.load

// CHECK-LABEL: tt.func @overlap_store(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK: tt.store
