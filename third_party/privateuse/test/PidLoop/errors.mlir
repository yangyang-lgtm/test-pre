// RUN: not privateuse-opt %s -privateuse-pid-loop='num-pids=8' -split-input-file 2>&1 | FileCheck %s
// RUN: not privateuse-opt %s -privateuse-pid-loop='num-pids=0' -split-input-file 2>&1 | FileCheck %s --check-prefix=NO_PIDS

// CHECK: error: pid-loop: missing module attribute privateuse.grid
// NO_PIDS: error: pid-loop: num-pids must be positive, got 0
module {
  func.func @kernel(%p: !tt.ptr<f32>) {
    return
  }
}

// -----

// CHECK: error: pid-loop: grid dimensions must be positive, got array<i64: 0, 1, 1>
module attributes {privateuse.grid = array<i64: 0, 1, 1>} {
  func.func @kernel(%p: !tt.ptr<f32>) {
    return
  }
}

// -----

// CHECK: error: pid-loop: expected a single-block kernel without results
module attributes {privateuse.grid = array<i64: 4, 1, 1>} {
  func.func @kernel() {
    cf.br ^bb1
  ^bb1:
    return
  }
}

// -----

// CHECK: error: pid-loop: expected a single-block kernel without results
module attributes {privateuse.grid = array<i64: 4, 1, 1>} {
  func.func @kernel() -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
