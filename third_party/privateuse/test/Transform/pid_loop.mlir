// RUN: privateuse-opt %s --privateuse-pid-loop='num-pids=32' -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @exact
// CHECK: %[[TID:.*]] = pumem.get_physical_id : index
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.store
// CHECK: return
module attributes {privateuse.grid = array<i64: 32>} {
  func.func @exact(%out: memref<32xi32>) {
    %pid = tt.get_program_id x : i32
    %idx = arith.index_cast %pid : i32 to index
    memref.store %pid, %out[%idx] : memref<32xi32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @guarded
// CHECK-DAG: %[[TID:.*]] = pumem.get_physical_id : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[TID]], %[[ONE]] : index
// CHECK: scf.if %[[COND]] {
// CHECK: memref.store
// CHECK: }
// CHECK-NOT: scf.for
module attributes {privateuse.grid = array<i64: 1>} {
  func.func @guarded(%out: memref<32xi32>) {
    %pid = tt.get_program_id x : i32
    %idx = arith.index_cast %pid : i32 to index
    memref.store %pid, %out[%idx] : memref<32xi32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @strided
// CHECK-DAG: %[[TOTAL:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 32 : index
// CHECK: %[[TID:.*]] = pumem.get_physical_id : index
// CHECK: scf.for %{{.*}} = %[[TID]] to %[[TOTAL]] step %[[STEP]]
// CHECK: memref.store
module attributes {privateuse.grid = array<i64: 64>} {
  func.func @strided(%out: memref<64xi32>) {
    %pid = tt.get_program_id x : i32
    %idx = arith.index_cast %pid : i32 to index
    memref.store %pid, %out[%idx] : memref<64xi32>
    return
  }
}
