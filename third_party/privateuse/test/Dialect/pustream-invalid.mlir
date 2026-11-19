// RUN: privateuse-opt %s -verify-diagnostics --split-input-file

func.func @slot_to_slot(%a: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %sram = memref.alloc() : memref<8xf32, 128>
  %s0 = pustream.slot %sram[%c0] : memref<8xf32, 128> -> !pustream.slot<memref<8xf32, 128>>
  %s1 = pustream.slot %sram[%c0] : memref<8xf32, 128> -> !pustream.slot<memref<8xf32, 128>>
  // expected-error @+1 {{exactly one operand must be a !pustream.slot; slot-to-slot and memref-to-memref copies are not expressible}}
  pustream.async.copy %s0, %s1 : !pustream.slot<memref<8xf32, 128>>, !pustream.slot<memref<8xf32, 128>>
  return
}

// -----

func.func @dram_to_dram(%a: memref<8xf32>, %b: memref<8xf32>) {
  // expected-error @+1 {{exactly one operand must be a !pustream.slot; slot-to-slot and memref-to-memref copies are not expressible}}
  pustream.async.copy %a, %b : memref<8xf32>, memref<8xf32>
  return
}

// -----

func.func @wait_without_producer() {
  %c0 = arith.constant 0 : index
  %sram = memref.alloc() : memref<8xf32, 128>
  %s = pustream.slot %sram[%c0] : memref<8xf32, 128> -> !pustream.slot<memref<8xf32, 128>>
  // expected-error @+1 {{waits a slot whose buffer never participates in a copy}}
  pustream.wait %s : !pustream.slot<memref<8xf32, 128>>
  return
}

// -----

func.func @stage_out_of_range() {
  %c5 = arith.constant 5 : index
  %in = memref.alloc() {pustream.stages = 4 : i32} : memref<4x256xf32, 128>
  // expected-error @+1 {{constant stage index 5 is outside [0, 4)}}
  %s = pustream.slot %in[%c5] : memref<4x256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
  return
}

// -----

func.func @dynamic_index_without_stages(%i: index) {
  %buf = memref.alloc() : memref<256xf32, 128>
  // expected-error @+1 {{a dynamic stage index requires a buffer with pustream.stages}}
  %s = pustream.slot %buf[%i] : memref<256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
  return
}

// -----

func.func @sync_not_last(%dram: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %sram = memref.alloc() : memref<8xf32, 128>
  %s = pustream.slot %sram[%c0] : memref<8xf32, 128> -> !pustream.slot<memref<8xf32, 128>>
  // expected-error @+1 {{must be the last statement of a kernel function}}
  pustream.sync
  pustream.async.copy %dram, %s : memref<8xf32>, !pustream.slot<memref<8xf32, 128>>
  return
}

// -----

func.func @dispatch_index_out_of_range(%a: memref<8xf32>) {
  // expected-error @+1 {{writes index 2 is out of range}}
  pustream.async.dispatch @k(%a) {reads = array<i64: 0>, writes = array<i64: 2>} : memref<8xf32>
  return
}

// -----

func.func @dispatch_scalar_effect(%a: memref<8xf32>, %s: f32) {
  // expected-error @+1 {{reads index 1 does not refer to a buffer argument}}
  pustream.async.dispatch @k(%a, %s) {reads = array<i64: 0, 1>, writes = array<i64>} : memref<8xf32>, f32
  return
}
