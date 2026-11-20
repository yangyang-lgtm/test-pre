// RUN: privateuse-opt %s -verify-diagnostics -split-input-file

// A statement that only orders the queues takes a token, not a memref.
func.func @wait_memref(%m: memref<256xf32, 128>) {
  // expected-error @+1 {{must be a !pustream.token}}
  %c = pustream.await %m : memref<256xf32, 128> -> !pustream.token
  return
}

// -----

// An allocation lives in SRAM and has a static shape.
func.func @alloc_dram() {
  // expected-error @+1 {{must live in SRAM}}
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32>
  return
}

// -----

// A transfer moves between spaces; buffer to buffer is copy.
func.func @transfer_both_sram() {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  // expected-error @+1 {{exactly one side must live in SRAM}}
  %x = pustream.transfer %t, %m, %m : !pustream.token, memref<256xf32, 128>, memref<256xf32, 128> -> !pustream.token
  return
}

// -----

// The token names the allocation the SRAM view belongs to.
func.func @transfer_wrong_allocation(%dram: memref<256xf32>) {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %o, %n = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  // expected-error @+1 {{not a view of this token's allocation}}
  %x = pustream.transfer %t, %dram, %n : !pustream.token, memref<256xf32>, memref<256xf32, 128> -> !pustream.token
  return
}

// -----

// A token with something in flight must be awaited before it is reused.
func.func @dirty_token(%dram: memref<256xf32>) {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %o, %n = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %l = pustream.transfer %t, %dram, %m : !pustream.token, memref<256xf32>, memref<256xf32, 128> -> !pustream.token
  // expected-error @+1 {{still in flight}}
  %c = pustream.copy %o, %n, %l, %m : !pustream.token, memref<256xf32, 128>, !pustream.token, memref<256xf32, 128> -> !pustream.token
  return
}

// -----

// pumem_wait(handle) cannot observe the compute queue.
func.func @await_dispatch() {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %o, %n = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %d = pustream.dispatch @body(%t, %m)(%o, %n)
     : (!pustream.token, memref<256xf32, 128>, !pustream.token, memref<256xf32, 128>)
       -> !pustream.token
  // expected-error @+1 {{compute queue}}
  %c = pustream.await %d : !pustream.token -> !pustream.token
  return
}

// -----

// pumem_wait_dispatch() cannot observe the copy executor.
func.func @await_compute_transfer(%dram: memref<256xf32>) {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %l = pustream.transfer %t, %dram, %m : !pustream.token, memref<256xf32>, memref<256xf32, 128> -> !pustream.token
  // expected-error @+1 {{copy executor}}
  %c = pustream.await_compute %l : !pustream.token -> !pustream.token
  return
}

// -----

// A dispatch writes its outputs; it does not read and write one allocation.
func.func @dispatch_in_place() {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  // expected-error @+1 {{share an allocation}}
  %d = pustream.dispatch @body(%t, %m)(%t, %m)
     : (!pustream.token, memref<256xf32, 128>, !pustream.token, memref<256xf32, 128>)
       -> !pustream.token
  return
}

// -----

// A copy needs two allocations.
func.func @copy_same_allocation() {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  // expected-error @+1 {{needs two allocations}}
  %c = pustream.copy %t, %m, %t, %m
    : !pustream.token, memref<256xf32, 128>, !pustream.token, memref<256xf32, 128> -> !pustream.token
  return
}

// -----

// A token input must be followed by the view it works on.
func.func @dispatch_unpaired() {
  %t, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %o, %n = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  // expected-error @+1 {{must be followed by the view}}
  %d = pustream.dispatch @body(%t, %t)(%o, %n)
     : (!pustream.token, !pustream.token, !pustream.token, memref<256xf32, 128>)
       -> !pustream.token
  return
}
