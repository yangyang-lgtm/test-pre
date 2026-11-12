// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-lower-unstructured-access \
// RUN:   -privateuse-ptr-to-memref -canonicalize | FileCheck %s

module attributes {privateuse.grid = array<i64: 4, 3, 1>} {
  tt.func @loop_base(%p: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %acc0 = tt.splat %cst : f32 -> tensor<8xf32>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%ptr = %a, %acc = %acc0) -> (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) : i32 {
      %v = tt.load %ptr : tensor<8x!tt.ptr<f32>>
      %sum = arith.addf %acc, %v : tensor<8xf32>
      %next = tt.addptr %ptr, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      scf.yield %next, %sum : tensor<8x!tt.ptr<f32>>, tensor<8xf32>
    }
    %pid = tt.get_program_id x : i32
    %c4 = arith.constant 4 : i32
    %cond = arith.cmpi slt, %pid, %c4 : i32
    %m = tt.splat %cond : i1 -> tensor<8xi1>
    %dS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %q = tt.addptr %dS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %q, %res#1, %m : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @volatile_load(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %a {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @ignore_cta_store(%dst: !tt.ptr<f32>) {
    %cst = arith.constant 1.000000e+00 : f32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %v = tt.splat %cst : f32 -> tensor<8xf32>
    %dS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %dS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %a, %v {ignore_cta} : tensor<8x!tt.ptr<f32>>
    tt.return
  }

}

// CHECK-LABEL: tt.func @loop_base(
// CHECK-SAME: %arg0: memref<?xf32>
// CHECK: scf.for
// CHECK: tt.load
// CHECK: scf.yield
// CHECK: pumem.scatter

// CHECK-LABEL: tt.func @volatile_load(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK-NOT: pumem.
// CHECK: tt.load

// CHECK-LABEL: tt.func @ignore_cta_store(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK-NOT: pumem.
// CHECK: tt.store
