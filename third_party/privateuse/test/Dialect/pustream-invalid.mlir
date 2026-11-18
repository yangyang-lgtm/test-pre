// RUN: privateuse-opt %s -verify-diagnostics --split-input-file

func.func @same_space_copy(%a: memref<8xf32, 128>, %b: memref<8xf32, 128>) {
  // expected-error @+1 {{requires source and target in different memory spaces}}
  %t = pustream.async.copy %a, %b : (memref<8xf32, 128>, memref<8xf32, 128>) -> !pustream.timepoint
  return
}

// -----

func.func @bad_dispatch_effect(%a: memref<8xf32>, %b: memref<8xf32, 128>) {
  // expected-error @+1 {{writes index 2 is out of range}}
  %t = pustream.async.dispatch @k(%a, %b) {reads = array<i64: 0>, writes = array<i64: 2>} : (memref<8xf32>, memref<8xf32, 128>) -> !pustream.timepoint
  return
}

// -----

func.func @scalar_dispatch_effect(%a: memref<8xf32>, %s: f32) {
  // expected-error @+1 {{reads index 1 does not refer to a memref argument}}
  %t = pustream.async.dispatch @k(%a, %s) {reads = array<i64: 0, 1>, writes = array<i64>} : (memref<8xf32>, f32) -> !pustream.timepoint
  return
}