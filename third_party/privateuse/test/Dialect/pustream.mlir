// RUN: privateuse-opt %s | FileCheck %s

// CHECK-LABEL: func.func @copy_dispatch(
func.func @copy_dispatch(%dram: memref<16x16xf32>, %sram: memref<16x16xf32, 128>, %acc: memref<16x16xf32, 128>, %beta: f32) {
  // CHECK: %[[IMM:.*]] = pustream.timepoint.immediate : !pustream.timepoint
  %imm = pustream.timepoint.immediate : !pustream.timepoint

  // CHECK: %[[CP:.*]] = pustream.async.copy %{{.*}}, %{{.*}} awaits(%[[IMM]]) : (memref<16x16xf32>, memref<16x16xf32, 128>, !pustream.timepoint) -> !pustream.timepoint
  %cp = pustream.async.copy %dram, %sram awaits(%imm) : (memref<16x16xf32>, memref<16x16xf32, 128>, !pustream.timepoint) -> !pustream.timepoint

  // CHECK: pustream.await %[[CP]] : !pustream.timepoint
  pustream.await %cp : !pustream.timepoint

  // CHECK: %[[DSP:.*]] = pustream.async.dispatch @k_step (%{{.*}}, %{{.*}}, %{{.*}}) awaits(%[[CP]]) {reads = array<i64: 0, 1>, writes = array<i64: 1>} : (memref<16x16xf32, 128>, memref<16x16xf32, 128>, f32, !pustream.timepoint) -> !pustream.timepoint
  %dsp = pustream.async.dispatch @k_step(%sram, %acc, %beta) awaits(%cp) {reads = array<i64: 0, 1>, writes = array<i64: 1>} : (memref<16x16xf32, 128>, memref<16x16xf32, 128>, f32, !pustream.timepoint) -> !pustream.timepoint

  // CHECK: %[[J:.*]] = pustream.timepoint.join(%[[CP]], %[[DSP]]) : (!pustream.timepoint, !pustream.timepoint) -> !pustream.timepoint
  %j = pustream.timepoint.join(%cp, %dsp) : (!pustream.timepoint, !pustream.timepoint) -> !pustream.timepoint

  // CHECK: pustream.await %[[J]] : !pustream.timepoint
  pustream.await %j : !pustream.timepoint
  return
}