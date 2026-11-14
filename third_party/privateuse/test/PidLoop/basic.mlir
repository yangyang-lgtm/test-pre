// RUN: privateuse-opt %s -privateuse-pid-loop='num-pids=8' -split-input-file | FileCheck %s

// A one-dimensional grid needs no division: the linear id is x.

// CHECK-LABEL: tt.func @flat
// CHECK: %[[TID:.*]] = pumem.get_physical_id : index
// CHECK: %[[TOTAL:.*]] = arith.constant 4 : index
// CHECK: %[[STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %[[L:.*]] = %[[TID]] to %[[TOTAL]] step %[[STEP]] {
// CHECK: %[[X:.*]] = arith.index_cast %[[L]] : index to i32
// CHECK: arith.muli %[[X]], %{{.*}} : i32
// CHECK-NOT: tt.get_program_id
// CHECK: tt.return
module attributes {privateuse.grid = array<i64: 4, 1, 1>} {
  tt.func @flat(%p: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %off = arith.muli %pid, %c2 : i32
    %v = tt.load %p : !tt.ptr<f32>
    tt.store %p, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// In two dimensions x = l % nx and y = l / nx, since there is no z axis.

// CHECK-LABEL: tt.func @grid_2d
// CHECK: scf.for %[[L2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[X2:.*]] = arith.remsi %[[L2]], %{{.*}} : index
// CHECK: %[[Q2:.*]] = arith.divsi %[[L2]], %{{.*}} : index
// CHECK: arith.index_cast %[[X2]] : index to i32
// CHECK: arith.index_cast %[[Q2]] : index to i32
// CHECK-NOT: tt.get_program_id
module attributes {privateuse.grid = array<i64: 4, 2, 1>} {
  tt.func @grid_2d(%p: !tt.ptr<f32>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %sx = arith.constant 4 : i32
    %sy = arith.constant 2 : i32
    %ox = arith.muli %x, %sx : i32
    %oy = arith.muli %y, %sy : i32
    %o = arith.addi %ox, %oy : i32
    %v = tt.load %p : !tt.ptr<f32>
    tt.store %p, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// In three dimensions z = (l / nx) / ny.

// CHECK-LABEL: tt.func @grid_3d
// CHECK: scf.for %[[L3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[X3:.*]] = arith.remsi %[[L3]], %{{.*}} : index
// CHECK: %[[Q3:.*]] = arith.divsi %[[L3]], %{{.*}} : index
// CHECK: %[[Y3:.*]] = arith.remsi %[[Q3]], %{{.*}} : index
// CHECK: %[[Z3:.*]] = arith.divsi %[[Q3]], %{{.*}} : index
// CHECK: arith.index_cast %[[X3]] : index to i32
// CHECK: arith.index_cast %[[Y3]] : index to i32
// CHECK: arith.index_cast %[[Z3]] : index to i32
// CHECK-NOT: tt.get_program_id
module attributes {privateuse.grid = array<i64: 2, 3, 4>} {
  tt.func @grid_3d(%p: !tt.ptr<f32>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %z = tt.get_program_id z : i32
    %sx = arith.constant 8 : i32
    %sy = arith.constant 4 : i32
    %ox = arith.muli %x, %sx : i32
    %oy = arith.muli %y, %sy : i32
    %oz = arith.addi %ox, %oy : i32
    %o = arith.addi %oz, %z : i32
    %v = tt.load %p : !tt.ptr<f32>
    tt.store %p, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// Kernels without program ids are still wrapped (the whole body runs once per
// program), while private helpers keep their program ids.

// CHECK-LABEL: func.func @no_pid
// CHECK: %[[TID4:.*]] = pumem.get_physical_id : index
// CHECK: scf.for %{{.*}} = %[[TID4]] to %{{.*}} step %{{.*}} {
// CHECK: tt.load
// CHECK: tt.store
// CHECK: return
// CHECK-LABEL: tt.func private @helper
// CHECK: tt.get_program_id x : i32
module attributes {privateuse.grid = array<i64: 2, 1, 1>} {
  func.func @no_pid(%p: !tt.ptr<f32>) {
    %v = tt.load %p : !tt.ptr<f32>
    tt.store %p, %v : !tt.ptr<f32>
    return
  }

  tt.func private @helper(%p: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %off = arith.muli %pid, %c0 : i32
    %v = tt.load %p : !tt.ptr<f32>
    tt.store %p, %v : !tt.ptr<f32>
    tt.return
  }
}
