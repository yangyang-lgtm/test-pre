#ifndef PUSTREAM_DIALECT_TD
#define PUSTREAM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def PuStream_Dialect : Dialect {
  let name = "pustream";

  let cppNamespace = "::mlir::pustream";

  let summary = "Async streaming and scheduling dialect";

  let description = [{
    `pustream` is the mid-level IR between bufferized linalg/memref compute
    and the `pumem` pointer ABI. It models asynchronous copies and dispatches
    with explicit completion tokens (timepoints) and hazard-based awaits.

    The dialect is backend-independent: the same IR is lowered to synchronous
    copies/calls for the CPU simulation backend and to DMA descriptors plus
    semaphores for the device backend.
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

def PuStream_TimepointType : PUSTREAM_Type<"Timepoint", "timepoint"> {
  let summary = "Completion token of an asynchronous operation";

  let description = [{
    A timepoint denotes the completion of one asynchronous operation or of a
    group joined by `pustream.timepoint.join`. It carries no payload; awaiting
    it is the only way to observe completion. The CPU backend erases all
    timepoints during lowering.

    Example: !pustream.timepoint
  }];
}

def PuStream_Timepoint : Type<CPred<"::llvm::isa<::mlir::pustream::TimepointType>($_self)">,
                              "a pustream timepoint", "::mlir::pustream::TimepointType">;

def PuStream_ImmediateOp : PUSTREAM_Op<"timepoint.immediate"> {
  let summary = "Create an already-completed timepoint";

  let description = [{
    Represents "no dependency". Used as the start point of a dependency chain
    and as the seed of loop-carried timepoints.

    Example: %t = pustream.timepoint.immediate : !pustream.timepoint
  }];
  let results = (outs PuStream_Timepoint:$done);
  let assemblyFormat = "attr-dict `:` type($done)";
}

def PuStream_AwaitOp : PUSTREAM_Op<"await"> {
  let summary = "Wait for an asynchronous operation to complete";

  let description = [{
    Placed at the point where the result of an asynchronous operation is
    actually consumed, never right after the operation is issued.

    Example: pustream.await %t : !pustream.timepoint
  }];
  let arguments = (ins PuStream_Timepoint:$timepoint);
  let assemblyFormat = "$timepoint attr-dict `:` type($timepoint)";
}

def PuStream_JoinOp : PUSTREAM_Op<"timepoint.join"> {
  let summary = "Join several timepoints into one";

  let description = [{
    The result completes when all inputs have completed.

    Example:
      %t = pustream.timepoint.join(%t0, %t1)
          : (!pustream.timepoint, !pustream.timepoint) -> !pustream.timepoint
  }];
  let arguments = (ins Variadic<PuStream_Timepoint>:$timepoints);
  let results = (outs PuStream_Timepoint:$done);
  let assemblyFormat = "`(` $timepoints `)` attr-dict `:` functional-type(operands, results)";
}

def PuStream_AsyncCopyOp : PUSTREAM_Op<"async.copy"> {
  let summary = "Asynchronously copy a strided box between DRAM and SRAM";

  let description = [{
    Copies the box described by `source` and `target` and returns a timepoint
    that completes when the transfer is done. Shape, strides, element type and
    dynamic offsets all come from the operands, so the op carries no redundant
    descriptor attributes.

    Only cross-space copies (DRAM <-> SRAM) are allowed; residual
    SRAM-to-SRAM copies must be rewritten before this op is formed.

    Example:
      %t = pustream.async.copy %dram, %sram
             : (memref<16x16xf32>, memref<16x16xf32, 128>)
               -> !pustream.timepoint
  }];
  let arguments = (ins AnyMemRef:$source,
                       AnyMemRef:$target,
                       Variadic<PuStream_Timepoint>:$awaits);
  let results = (outs PuStream_Timepoint:$done);
  let assemblyFormat = [{
    $source `,` $target (`awaits` `(` $awaits^ `)`)? attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def PuStream_AsyncDispatchOp : PUSTREAM_Op<"async.dispatch", [AttrSizedOperandSegments]> {
  let summary = "Asynchronously dispatch an outlined executable or a runtime function";

  let description = [{
    Dispatches `callee` with the given operands. Operands are memrefs or
    scalars and keep the callee's parameter order. `reads` and `writes` are
    indices into `args` and record the memref effects of the executable; the
    body is opaque after formation. Scalars are passed by value, so they never
    appear in `reads` or `writes` and do not participate in hazards. The result
    timepoint completes when the dispatch (and its writes) are done.

    `templateArgs` is only set when `callee` is an external runtime function
    (e.g. `runtime_matmul`). Each entry is a type or a compile-time value;
    the CPU lowering turns such a dispatch into `pumem.runtime_call`, which
    the C emitter prints as `callee<templates>(...)`.

    Examples:
      %t = pustream.async.dispatch @k_step(%a, %b, %acc, %beta)
             {reads = array<i64: 0, 1, 2>, writes = array<i64: 2>}
             : (memref<128x32xf32, 128>, memref<32x64xf32, 128>,
                memref<128x64xf32, 128>, f32) -> !pustream.timepoint
      %t = pustream.async.dispatch @runtime_matmul<[f32, 64 : index, 64 : index, 64 : index]>(%a, %b, %c)
             {reads = array<i64: 0, 1, 2>, writes = array<i64: 2>}
             : (memref<64x64xf32, 128>, memref<64x64xf32, 128>,
                memref<64x64xf32, 128>) -> !pustream.timepoint
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<ArrayAttr>:$templateArgs,
                       Variadic<AnyType>:$args,
                       Variadic<PuStream_Timepoint>:$awaits,
                       DenseI64ArrayAttr:$reads,
                       DenseI64ArrayAttr:$writes);
  let results = (outs PuStream_Timepoint:$done);
  let assemblyFormat = [{
    $callee (`<` $templateArgs^ `>`)? `(` $args `)` (`awaits` `(` $awaits^ `)`)? attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // PUSTREAM_DIALECT_TD