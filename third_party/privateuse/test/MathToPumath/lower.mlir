// RUN: privateuse-opt %s -math-to-pumath | FileCheck %s \
// RUN:   --implicit-check-not="math."

// Every math op is rewritten 1:1 into pumath, keeping the operand and result
// types.

// CHECK-LABEL: func.func @math(
// CHECK-SAME:    %[[A:.*]]: f32, %[[B:.*]]: f32, %[[I:.*]]: i32) -> (f32, f32, i32) {
// CHECK:         %[[EXP:.*]] = pumath.exp %[[A]] : f32
// CHECK:         %[[POW:.*]] = pumath.powf %[[A]], %[[B]] : f32
// CHECK:         %[[IPOW:.*]] = pumath.ipowi %[[I]], %[[I]] : i32
// CHECK:         return %[[EXP]], %[[POW]], %[[IPOW]] : f32, f32, i32
func.func @math(%a: f32, %b: f32, %i: i32) -> (f32, f32, i32) {
  %e = math.exp %a : f32
  %p = math.powf %a, %b : f32
  %r = math.ipowi %i, %i : i32
  return %e, %p, %r : f32, f32, i32
}
