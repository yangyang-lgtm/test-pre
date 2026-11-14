// RUN: privateuse-opt %s -privateuse-sink-constant-call-args | FileCheck %s

// Argument positions that receive the same constant from every call site are
// removed from the signature and rebuilt inside the callee. Non-constant
// arguments, and positions where the call sites disagree, stay in place.

// CHECK-LABEL: func.func private @callee(
// CHECK-SAME:    %[[KEEP:[a-zA-Z0-9_]+]]: index, %[[SCALE:[a-zA-Z0-9_]+]]: index) -> index {
// CHECK-DAG:     %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : index
// CHECK-DAG:     %[[ONE:[a-zA-Z0-9_]+]] = arith.constant 1 : index
// CHECK-DAG:     %[[SIZE:[a-zA-Z0-9_]+]] = arith.constant 64 : index
// CHECK:         %{{.*}} = arith.muli %[[KEEP]], %[[ZERO]] : index
// CHECK:         %{{.*}} = arith.addi %{{.*}}, %[[ONE]] : index
// CHECK:         %{{.*}} = arith.muli %{{.*}}, %[[SIZE]] : index
// CHECK:         %{{.*}} = arith.muli %{{.*}}, %[[SCALE]] : index
// CHECK:         return
//
// CHECK-LABEL: func.func @caller(
// CHECK:         call @callee(%[[N:[a-zA-Z0-9_]+]], %[[N]]) : (index, index) -> index

func.func private @callee(%keep: index, %zero: index, %one: index, %size: index, %scale: index) -> index {
  %a = arith.muli %keep, %zero : index
  %b = arith.addi %a, %one : index
  %c = arith.muli %b, %size : index
  %d = arith.muli %c, %scale : index
  return %d : index
}

func.func @caller(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %r = call @callee(%n, %c0, %c1, %c64, %n) : (index, index, index, index, index) -> index
  return %r : index
}

// The two call sites pass different constants for %v, so it stays an argument.

// CHECK-LABEL: func.func private @two_calls(%{{.*}}: index, %{{.*}}: index) -> index {
// CHECK:         arith.muli
//
// CHECK-LABEL: func.func @caller_a(
// CHECK:         call @two_calls({{.*}}) : (index, index) -> index
//
// CHECK-LABEL: func.func @caller_b(
// CHECK:         call @two_calls({{.*}}) : (index, index) -> index

func.func private @two_calls(%keep: index, %v: index) -> index {
  %r = arith.muli %keep, %v : index
  return %r : index
}

func.func @caller_a(%n: index) -> index {
  %c1 = arith.constant 1 : index
  %r = call @two_calls(%n, %c1) : (index, index) -> index
  return %r : index
}

func.func @caller_b(%n: index) -> index {
  %c2 = arith.constant 2 : index
  %r = call @two_calls(%n, %c2) : (index, index) -> index
  return %r : index
}
