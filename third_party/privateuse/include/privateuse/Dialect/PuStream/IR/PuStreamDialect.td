#ifndef PUSTREAM_DIALECT_TD
#define PUSTREAM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PuStream_Dialect : Dialect {
  let name = "pustream";
  let cppNamespace = "::mlir::pustream";
  let summary = "Statement-form async streaming dialect";
  let description = [{
    `pustream` sits between bufferized linalg/memref compute and the `pumem`
    pointer ABI. It describes an **issue sequence**, not a dependency graph: the
    async statements mirror the runtime statements one to one, carry no results,
    and are totally ordered by an ordering side-effect resource.

    The device contract behind that shape:

      * the host thread is the only issuer and the only thing that orders work
        across executors;
      * each copy direction is a global FIFO with a single consumer, so copies of
        the same direction can never run concurrently;
      * the compute executor is one structure with several lanes. It exposes no
        per-task completion value; the only observable completion is a global
        barrier that drains every issued dispatch.

    Consequences encoded here:

      * no completion tokens. A DMA handle names a *slot*, not an operation, so
        two copies into the same slot share one completion event and a token
        could not be lowered. Awaiting a subset of dispatches is not expressible
        either.
      * `wait` takes a `!pustream.slot`, a stage-qualified SRAM buffer, because
        only a slot has a handle to wait on.
      * a DMA copy is exactly `(src, dst)` with one side a slot; direction follows
        from which side is staged, so a wrongly-directed copy cannot be written.

    See docs/pustream-v4-design.md.
  }];
  let useDefaultTypePrinterParser = 1;
}

class PUSTREAM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PuStream_Dialect, mnemonic, traits> {
}

class PUSTREAM_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<PuStream_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Ordering
//===----------------------------------------------------------------------===//

// One resource for the whole async statement sequence. The host issues in program
// order and each copy direction is a total order, so no pass may reorder two async
// statements. Making that a side effect removes the "a later pass moved my wait"
// class of bugs instead of leaving it to pass discipline.
def PuStream_OrderingRes : Resource<"::mlir::pustream::Ordering">;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def PuStream_SlotType : PUSTREAM_Type<"Slot", "slot"> {
  let parameters = (ins "::mlir::MemRefType":$tile);
  // Custom, so a slot prints in full (!pustream.slot<...>); see the definitions
  // in PuStreamDialect.cpp.
  let hasCustomAssemblyFormat = 1;
  let summary = "A stage-qualified SRAM slot";
  let description = [{
    A `!pustream.slot<T>` value denotes one stage of a staged SRAM buffer: the
    value carries `(base alloc, runtime index)`, the type carries the tile shape
    `T`. It is the only operand type accepted by `pustream.wait` and the only
    SRAM-side operand type accepted by `pustream.async.copy`, so an op that needs
    a handle can always name one and an op that has none cannot pretend to.

    Example: !pustream.slot<memref<256xf32, 128>>
  }];
}

def PuStream_Slot : Type<CPred<"::llvm::isa<::mlir::pustream::SlotType>($_self)">,
                          "a pustream stage slot", "::mlir::pustream::SlotType">;

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

def PuStream_SlotOp : PUSTREAM_Op<"slot", [Pure]> {
  let summary = "View one stage of a staged SRAM buffer";
  let description = [{
    `buffer` is either a staged buffer carrying `pustream.stages` (dim 0 is the
    stage dimension, the tile type is the remaining dims) or a plain single-stage
    SRAM buffer (the tile type equals the buffer type, index must be 0). The op is
    `Pure`: it only builds a descriptor, so CSE and LICM move it freely. Every
    async statement, by contrast, keeps its position.

    Example:
      %s = pustream.slot %in[%i]
             : memref<4x256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
  }];
  let arguments = (ins AnyMemRef:$buffer, Index:$index);
  let results = (outs PuStream_Slot:$slot);
  let assemblyFormat = "$buffer `[` $index `]` attr-dict `:` type($buffer) `->` type($slot)";
  let hasVerifier = 1;
}

def PuStream_AsyncCopyOp : PUSTREAM_Op<"async.copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Enqueue a DRAM <-> SRAM transfer on one slot";
  let description = [{
    Exactly one operand is a `!pustream.slot`: the staged side. The other is a
    DRAM memref, which may have a dynamic extent and dynamic strides (partial
    tiles). The box geometry comes from the DRAM operand; the slot contributes
    only base address and capacity. Direction is static: a slot target is
    `dram_to_sram`, a slot source is `sram_to_dram`. Slot-to-slot and
    memref-to-memref forms are rejected, which makes the direction invariant
    unexpressible rather than checked.

    Example:
      pustream.async.copy %dram_slice, %s
        : memref<?xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
  }];
  let arguments = (ins AnyType:$source, AnyType:$target);
  // Custom, so a slot operand prints in full (!pustream.slot<...>). Letting
  // TableGen print it double-wraps the type on this MLIR version.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PuStream_SyncCopyOp : PUSTREAM_Op<"sync.copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy a strided block between two SRAM buffers, synchronously";

  let description = [{
    Both operands are SRAM memrefs. This is the statement form of a host copy: it
    blocks until the data is in place, so there is no slot, no handle and nothing
    to wait on. Same shape as the async copy statement, minus the staging.

    Example:
      pustream.sync.copy %acc_tile, %acc : memref<f32, 128>, memref<f32, 128>
  }];
  let arguments = (ins AnyType:$source, AnyType:$target);
  let assemblyFormat = "$source `,` $target attr-dict `:` type($source) `,` type($target)";
  let hasVerifier = 1;
}
def PuStream_AsyncDispatchOp : PUSTREAM_Op<"async.dispatch", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Submit an outlined executable to the compute executor";
  let description = [{
    Operands keep the callee's parameter order and may be slots, memrefs or
    scalars. `reads` and `writes` are indices into `args` and are all the barrier
    pass needs: it asks "does this DMA touch a buffer this dispatch read or",
    never "which kind of hazard is it". The op returns nothing: the compute
    executor has no per-task completion value, and the only way to observe
    completion is `pustream.wait_dispatch`.

    Example:
      pustream.async.dispatch @k_step(%pin, %pout, %beta)
        {reads = array<i64: 0, 2>, writes = array<i64: 1>}
        : !pustream.slot<memref<256xf32, 128>>, !pustream.slot<memref<256xf32, 128>>, f32
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<ArrayAttr>:$templateArgs,
                       Variadic<AnyType>:$args,
                       DenseI64ArrayAttr:$reads,
                       DenseI64ArrayAttr:$writes);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PuStream_WaitOp : PUSTREAM_Op<"wait", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Block until a slot has no DMA in flight";
  let description = [{
    Waits the handle of `slot`: the last enqueued transfer in *each* direction,
    skipping a direction with nothing recorded. Since each direction is a global
    FIFO, waiting the tail implies the whole prefix retired, so this one statement
    covers both the RAW case (a consumer waits for its load) and the WAR case (a
    producer waits for its reader). A direction-selective wait is deliberately not
    expressible: the handle keeps only two tail pointers, so the distinction would
    not be reliable.

    Example: pustream.wait %pin
  }];
  let arguments = (ins PuStream_Slot:$slot);
  let assemblyFormat = "$slot attr-dict `:` type($slot)";
  let hasVerifier = 1;
}

def PuStream_WaitDispatchOp : PUSTREAM_Op<"wait_dispatch", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Block until every issued dispatch has retired";
  let description = [{
    The compute executor's only observable completion: a global, idempotent
    barrier. It takes no operands because the backend cannot wait for a subset of
    dispatches; naming them would promise something the runtime cannot do.

    Example: pustream.wait_dispatch
  }];
  let assemblyFormat = "attr-dict";
}

def PuStream_SyncOp : PUSTREAM_Op<"sync", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Drain every queue, at kernel exit";
  let description = [{
    Drains copies and dispatches alike. Only legal as the last statement of a
    kernel function, which the verifier enforces so the exit-drain invariant has
    a single place to check.

    Example: pustream.sync
  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // PUSTREAM_DIALECT_TD
