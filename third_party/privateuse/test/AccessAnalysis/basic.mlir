// RUN: privateuse-opt %s -privateuse-analyze-access -o /dev/null 2>&1 | FileCheck %s

// CHECK: [privateuse-access] op 1: tt.load rank=0 elem_bytes=4 offset=0 dims=[] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 2: tt.load rank=0 elem_bytes=4 offset=0 dims=[] mask=scalar deferred=none
// CHECK-NEXT: [privateuse-access] op 3: tt.load rank=0 elem_bytes=4 offset=0 dims=[] mask=all_true deferred=none
// CHECK-NEXT: [privateuse-access] op 4: tt.store rank=0 elem_bytes=4 offset=0 dims=[] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 5: tt.store rank=0 elem_bytes=4 offset=0 dims=[] mask=scalar deferred=none
// CHECK-NEXT: [privateuse-access] op 6: tt.load rank=1 elem_bytes=4 offset=(pid.x * 128) dims=[dim0={start:0,stride:1,size:128}] mask=all_true deferred=none
// CHECK-NEXT: [privateuse-access] op 7: tt.store rank=1 elem_bytes=1 offset=0 dims=[dim0={start:0,stride:1,size:128}] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 8: tt.load rank=1 elem_bytes=4 offset=0 dims=[dim0={start:0,stride:1,size:64}] mask=all_false deferred=none
// CHECK-NEXT: [privateuse-access] op 9: tt.store rank=1 elem_bytes=4 offset=0 dims=[dim0={start:0,stride:1,size:64}] mask=all_true deferred=none
// CHECK-NEXT: [privateuse-access] op 10: tt.load rank=2 elem_bytes=4 offset=0 dims=[dim0={start:0,stride:64,size:64}, dim1={start:0,stride:1,size:64}] mask=box dim1=[((0 - (pid.x + 16)) + 1*i < 0)] deferred=none
// CHECK-NEXT: [privateuse-access] op 11: tt.load rank=3 elem_bytes=4 offset=0 dims=[dim0={start:0,stride:512,size:8}, dim1={start:0,stride:32,size:16}, dim2={start:0,stride:1,size:32}] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 12: tt.load rank=1 elem_bytes=4 offset=0 dims=[dim0={start:5,stride:0,size:8}] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 13: tt.load rank=1 elem_bytes=4 offset=0 dims=[dim0={start:0,stride:1,size:128}] mask=none deferred=none
// CHECK-NEXT: [privateuse-access] op 14: tt.load rank=1 elem_bytes=4 offset=(pid.x * 32) dims=[dim0={start:0,stride:1,size:128}] mask=all_true deferred=none
// CHECK-NEXT: [privateuse-access] summary ops=14 loads=10 stores=4 atomics=0 descriptors=0 deferred=0 errors=0 reasons={}

module attributes {privateuse.grid = array<i64: 4, 3, 1>} {
  tt.func @scalar(%p: !tt.ptr<f32>, %m: i1) {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %v = tt.load %p : !tt.ptr<f32>
    %v2 = tt.load %p, %m, %cst : !tt.ptr<f32>
    %v3 = tt.load %p, %true, %cst : !tt.ptr<f32>
    tt.store %p, %v : !tt.ptr<f32>
    tt.store %p, %v2, %m : !tt.ptr<f32>
    tt.return
  }

  tt.func @pid_range(%p: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c128 = arith.constant 128 : i32
    %pidOff = arith.muli %pid, %c128 : i32
    %pidS = tt.splat %pidOff : i32 -> tensor<128xi32>
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %idx = arith.addi %pidS, %r : tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %a = tt.addptr %pS, %idx : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %c128S = tt.splat %c128 : i32 -> tensor<128xi32>
    %full = arith.cmpi slt, %r, %c128S : tensor<128xi32>
    %v = tt.load %a, %full : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @ptr_bitcast(%p: !tt.ptr<i1>) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %a = tt.addptr %pS, %r : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %b = tt.bitcast %a : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %v = arith.constant dense<1> : tensor<128xi8>
    tt.store %b, %v : tensor<128x!tt.ptr<i8>>
    tt.return
  }

  tt.func @mask_static(%p: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %c0S = tt.splat %c0 : i32 -> tensor<64xi32>
    %c64S = tt.splat %c64 : i32 -> tensor<64xi32>
    %never = arith.cmpi slt, %r, %c0S : tensor<64xi32>
    %always = arith.cmpi slt, %r, %c64S : tensor<64xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %a = tt.addptr %pS, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %a, %never : tensor<64x!tt.ptr<f32>>
    tt.store %a, %v, %always : tensor<64x!tt.ptr<f32>>
    tt.return
  }

  tt.func @box_mask_2d(%p: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %bound = arith.addi %pid, %c16 : i32
    %c64 = arith.constant 64 : i32
    %rows = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %cols = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %rowsE = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %c64S1 = tt.splat %c64 : i32 -> tensor<64x1xi32>
    %rowOff = arith.muli %rowsE, %c64S1 : tensor<64x1xi32>
    %rowBc = tt.broadcast %rowOff : tensor<64x1xi32> -> tensor<64x64xi32>
    %colsE = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %colsBc = tt.broadcast %colsE : tensor<1x64xi32> -> tensor<64x64xi32>
    %idx = arith.addi %rowBc, %colsBc : tensor<64x64xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %a = tt.addptr %pS, %idx : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %boundS = tt.splat %bound : i32 -> tensor<64x64xi32>
    %mask = arith.cmpi slt, %colsBc, %boundS : tensor<64x64xi32>
    %v = tt.load %a, %mask : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }

  tt.func @dims3d(%p: !tt.ptr<f32>) {
    %r0 = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %r1 = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %c512 = arith.constant 512 : i32
    %c32 = arith.constant 32 : i32
    %r0e1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %r0e2 = tt.expand_dims %r0e1 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %c512S = tt.splat %c512 : i32 -> tensor<8x1x1xi32>
    %t0 = arith.muli %r0e2, %c512S : tensor<8x1x1xi32>
    %b0 = tt.broadcast %t0 : tensor<8x1x1xi32> -> tensor<8x16x32xi32>
    %r1e1 = tt.expand_dims %r1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %r1e2 = tt.expand_dims %r1e1 {axis = 2 : i32} : tensor<1x16xi32> -> tensor<1x16x1xi32>
    %c32S = tt.splat %c32 : i32 -> tensor<1x16x1xi32>
    %t1 = arith.muli %r1e2, %c32S : tensor<1x16x1xi32>
    %b1 = tt.broadcast %t1 : tensor<1x16x1xi32> -> tensor<8x16x32xi32>
    %r2e1 = tt.expand_dims %r2 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %r2e2 = tt.expand_dims %r2e1 {axis = 1 : i32} : tensor<1x32xi32> -> tensor<1x1x32xi32>
    %b2 = tt.broadcast %r2e2 : tensor<1x1x32xi32> -> tensor<8x16x32xi32>
    %s01 = arith.addi %b0, %b1 : tensor<8x16x32xi32>
    %idx = arith.addi %s01, %b2 : tensor<8x16x32xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x16x32x!tt.ptr<f32>>
    %a = tt.addptr %pS, %idx : tensor<8x16x32x!tt.ptr<f32>>, tensor<8x16x32xi32>
    %v = tt.load %a : tensor<8x16x32x!tt.ptr<f32>>
    tt.return
  }

  tt.func @broadcast_start(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 5 : i32, end = 6 : i32} : tensor<1xi32>
    %bc = tt.broadcast %r : tensor<1xi32> -> tensor<8xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %bc : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %a : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @reshape_address(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %e = tt.reshape %r : tensor<128xi32> -> tensor<1x128xi32>
    %c = tt.reshape %e : tensor<1x128xi32> -> tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %a = tt.addptr %pS, %c : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %a : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @mask_negative_stride(%p: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c32 = arith.constant 32 : i32
    %pidOff = arith.muli %pid, %c32 : i32
    %pidS = tt.splat %pidOff : i32 -> tensor<128xi32>
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %idx = arith.addi %pidS, %r : tensor<128xi32>
    %zero = arith.constant dense<0> : tensor<128xi32>
    %mask = arith.cmpi sge, %idx, %zero : tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %a = tt.addptr %pS, %idx : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %a, %mask : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
