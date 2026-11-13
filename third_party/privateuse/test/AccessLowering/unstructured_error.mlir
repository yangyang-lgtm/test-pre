// RUN: privateuse-opt %s -privateuse-lower-unstructured-access -verify-diagnostics

// lower-or-error 契约：每个 tt.load/tt.store 要么降成 gather/scatter，要么硬报错。
// 白名单门：访存 op 只放行 tt.load/tt.store，descriptor/atomic op 一律报错。
// 本文件单独跑 unstructured，逐门验证报错诊断（循环 base / volatile / ignore_cta /
// 标量 / i1 直访 / reshape 指针链 / atomic / descriptor）。

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
      // expected-error @+1 {{pointer base is not a kernel entry-block argument}}
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
    // expected-error @+1 {{volatile load cannot be lowered to gather/scatter}}
    %v = tt.load %a {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @ignore_cta_store(%dst: !tt.ptr<f32>) {
    %cst = arith.constant 1.000000e+00 : f32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %v = tt.splat %cst : f32 -> tensor<8xf32>
    %dS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %dS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    // expected-error @+1 {{ignore_cta store cannot be lowered to gather/scatter}}
    tt.store %a, %v {ignore_cta} : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @dynamic_step(%p: !tt.ptr<f32>, %n: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %acc0 = tt.splat %cst : f32 -> tensor<128xf32>
    %stepS = tt.splat %step : i32 -> tensor<128xi32>
    %res:2 = scf.for %iv = %c0 to %n step %step iter_args(%ptr = %pS, %acc = %acc0) -> (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) : i32 {
      // expected-error @+1 {{pointer base is not a kernel entry-block argument}}
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

  tt.func @loop_carried(%p: !tt.ptr<f32>, %n: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %c4S = tt.splat %c4 : i32 -> tensor<8xi32>
    %off = tt.addptr %pS, %c4S : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %acc0 = tt.splat %cst : f32 -> tensor<8xf32>
    %cond = arith.cmpi slt, %pid, %c4 : i32
    %half = tt.splat %cond : i1 -> tensor<8xi1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%ptr = %off, %acc = %acc0) -> (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) : i32 {
      // expected-error @+1 {{pointer base is not a kernel entry-block argument}}
      %v = tt.load %ptr, %half : tensor<8x!tt.ptr<f32>>
      %sum = arith.addf %acc, %v : tensor<8xf32>
      %next = tt.addptr %ptr, %c4S : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      scf.yield %next, %sum : tensor<8x!tt.ptr<f32>>, tensor<8xf32>
    }
    %q = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %q, %res#1 : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @scalar_access(%p: !tt.ptr<f32>) {
    // expected-error @+1 {{access is not a ranked pointer tensor}}
    %v = tt.load %p : !tt.ptr<f32>
    tt.store %p, %v : !tt.ptr<f32>
    tt.return
  }

  tt.func @i1_direct(%p: !tt.ptr<i1>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<i1> -> tensor<8x!tt.ptr<i1>>
    %a = tt.addptr %pS, %r : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
    // expected-error @+1 {{direct i1 access requires i8 storage packing}}
    %v = tt.load %a : tensor<8x!tt.ptr<i1>>
    tt.store %a, %v : tensor<8x!tt.ptr<i1>>
    tt.return
  }

  tt.func @reshape_chain(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %q = tt.reshape %a : tensor<8x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    // expected-error @+1 {{pointer base is not a kernel entry-block argument}}
    %v = tt.load %q : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }

  tt.func @atomic_rmw(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %c = arith.constant 1.000000e+00 : f32
    %cS = tt.splat %c : f32 -> tensor<8xf32>
    // expected-error @+1 {{unsupported access op}}
    %v = tt.atomic_rmw fadd, acq_rel, gpu, %a, %cS : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) -> tensor<8xf32>
    tt.return
  }

  tt.func @descriptor_load(%d: !tt.tensordesc<1x128x64xf16>) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{unsupported access op}}
    %l = tt.descriptor_load %d[%c0, %c0, %c0] : !tt.tensordesc<1x128x64xf16> -> tensor<1x128x64xf16>
    tt.return
  }
}