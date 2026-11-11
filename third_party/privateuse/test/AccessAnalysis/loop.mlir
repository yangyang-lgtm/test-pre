// RUN: privateuse-opt %s -privateuse-analyze-access -verify-diagnostics -o /dev/null 2>&1 | FileCheck %s

// CHECK: [privateuse-access] op 1: tt.load rank=1 elem_bytes=4 offset=((pid.x * 128) + (iv0 * 32)) dims=[dim0={start:0,stride:1,size:128}] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] summary ops=2 loads=2 stores=0 atomics=0 descriptors=0 deferred=0 errors=1 reasons={}

module attributes {privateuse.grid = array<i64: 4, 1, 1>} {
  tt.func @loop_carried(%p: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %pid = tt.get_program_id x : i32
    %pidOff = arith.muli %pid, %c128 : i32
    %pidS = tt.splat %pidOff : i32 -> tensor<128xi32>
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %idx = arith.addi %pidS, %r : tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %init = tt.addptr %pS, %idx : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %acc0 = tt.splat %cst : f32 -> tensor<128xf32>
    %stepS = tt.splat %c32 : i32 -> tensor<128xi32>
    %res:2 = scf.for %iv = %c0 to %n step %c32 iter_args(%ptr = %init, %acc = %acc0) -> (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) : i32 {
      %v = tt.load %ptr : tensor<128x!tt.ptr<f32>>
      %sum = arith.addf %acc, %v : tensor<128xf32>
      %next = tt.addptr %ptr, %stepS : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      scf.yield %next, %sum : tensor<128x!tt.ptr<f32>>, tensor<128xf32>
    }
    tt.return
  }

  tt.func @loop_dynamic_step(%p: !tt.ptr<f32>, %n: i32, %step: i32) { // expected-error {{unsupported block argument in address expression}}
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
    tt.return
  }
}
