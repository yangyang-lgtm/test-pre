// RUN: privateuse-opt %s -privateuse-analyze-access -verify-diagnostics -o /dev/null 2>&1 | FileCheck %s

// CHECK: [privateuse-access] op 1: tt.load rank=1 elem_bytes=4 offset=127 dims=[dim0={start:0,stride:-1,size:128}] mask=none deferred=negative_stride
// CHECK-NEXT: [privateuse-access] op 2: tt.atomic_rmw rank=? elem_bytes=? offset=? dims=[] mask=? deferred=atomic
// CHECK-NEXT: [privateuse-access] op 3: tt.descriptor_load rank=? elem_bytes=? offset=? dims=[] mask=? deferred=unsupported_descriptor
// CHECK-NEXT: [privateuse-access] op 4: tt.gather rank=? elem_bytes=? offset=? dims=[] mask=? deferred=data_dependent_offset
// CHECK-NEXT: [privateuse-access] op 6: tt.load rank=0 elem_bytes=8 offset=0 dims=[] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 8: tt.load rank=? elem_bytes=? offset=? dims=[] mask=? deferred=data_dependent_offset
// CHECK-NEXT: [privateuse-access] op 9: tt.load rank=1 elem_bytes=4 offset=0 dims=[dim0={start:0,stride:0,size:128}] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 10: tt.load rank=? elem_bytes=? offset=? dims=[] mask=? deferred=data_dependent_offset
// CHECK-NEXT: [privateuse-access] summary ops=11 loads=8 stores=0 atomics=1 descriptors=1 deferred=6 errors=3 reasons={negative_stride=1,data_dependent_offset=3,unsupported_descriptor=1,atomic=1}

module attributes {privateuse.grid = array<i64: 4, 1, 1>} {
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

  tt.func @atomic(%p: !tt.ptr<i32>, %v: i32) {
    %old = tt.atomic_rmw add, relaxed, gpu, %p, %v : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }

  tt.func @descriptor(%d: !tt.tensordesc<1x128x64xf16>) {
    %c0 = arith.constant 0 : i32
    %l = tt.descriptor_load %d[%c0, %c0, %c0] : !tt.tensordesc<1x128x64xf16> -> tensor<1x128x64xf16>
    tt.return
  }

  tt.func @gather(%src: tensor<128x16xf32>, %idx: tensor<512x16xi32>) {
    %g = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x16xi32>) -> tensor<512x16xf32>
    tt.return
  }

  tt.func @coupled_mask(%p: !tt.ptr<f32>) {
    %r0 = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r1 = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r0e = tt.expand_dims %r0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %r0b = tt.broadcast %r0e : tensor<64x1xi32> -> tensor<64x64xi32>
    %r1e = tt.expand_dims %r1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %r1b = tt.broadcast %r1e : tensor<1x64xi32> -> tensor<64x64xi32>
    %mask = arith.cmpi slt, %r0b, %r1b : tensor<64x64xi32> // expected-error {{mask couples several dimensions}}
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %v = tt.load %pS, %mask : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }

  tt.func @pointer_from_load(%pp: !tt.ptr<!tt.ptr<f32>>) {
    %p2 = tt.load %pp : !tt.ptr<!tt.ptr<f32>> // expected-error {{pointer is loaded from memory}}
    %v = tt.load %p2 : !tt.ptr<f32>
    tt.return
  }

  tt.func @iv_dependent_step(%p: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %res = scf.for %iv = %c0 to %n step %c1 iter_args(%ptr = %pS) -> (tensor<128x!tt.ptr<f32>>) : i32 {
      %v = tt.load %ptr : tensor<128x!tt.ptr<f32>>
      %ivS = tt.splat %iv : i32 -> tensor<128xi32>
      %next = tt.addptr %ptr, %ivS : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      scf.yield %next : tensor<128x!tt.ptr<f32>>
    }
    tt.return
  }

  tt.func @loaded_offset(%p: !tt.ptr<f32>, %idx: !tt.ptr<i32>) {
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %idxS = tt.splat %idx : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %i = tt.load %idxS : tensor<128x!tt.ptr<i32>>
    %q = tt.addptr %pS, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %q : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @reshape_reject(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %v = tt.reshape %r : tensor<128xi32> -> tensor<64x2xi32> // expected-error {{reshape changes the address layout}}
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<64x2x!tt.ptr<f32>>
    %a = tt.addptr %pS, %v : tensor<64x2x!tt.ptr<f32>>, tensor<64x2xi32>
    %val = tt.load %a : tensor<64x2x!tt.ptr<f32>>
    tt.return
  }
}
