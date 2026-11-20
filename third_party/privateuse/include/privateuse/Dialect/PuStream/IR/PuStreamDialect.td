#ifndef PUSTREAM_DIALECT_TD
#define PUSTREAM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PuStream_Dialect : Dialect {
  let name = "pustream";
  let cppNamespace = "::mlir::pustream";

  let summary = "The asynchronous statement layer between linalg and the runtime";

  let description = [{
    The runtime can wait on exactly two things: one allocation's sequence point
    (pumem_wait(handle)) and the compute queue up to the moment of the call
    (pumem_wait_dispatch). There is no per-operation completion value, so there
    is no timepoint type here: readiness belongs to an allocation, and the two
    waits are two statements.

    An allocation is therefore split in two. A `!pustream.token` carries the
    identity and the readiness; a plain memref carries the shape, and views are
    ordinary memref.subview of it. Statements that only order the queues take
    the token, statements that move data take the token and the view together.

    Every async statement returns the token it wrote, so a token's state travels
    in SSA. The value a transfer, copy or dispatch produces is dirty (something
    is in flight over that allocation) and may only be consumed by await or
    await_compute, which hand back the clean token. Using a dirty token directly
    is what makes the runtime abort on a busy descriptor, so the verifier
    rejects it instead.

    Statement order is the issue order. Two statements over different
    allocations with no await in between are both in flight, which is exactly
    how the copy and compute executors overlap.
  }];

  let useDefaultTypePrinterParser = 1;
}

class PUSTREAM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PuStream_Dialect, mnemonic, traits>;

class PUSTREAM_Type<string name, string typeMnemonic> :
    TypeDef<PuStream_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def PuStream_TokenType : PUSTREAM_Type<"Token", "token"> {
  // mlir::TokenType is a builtin type, so the C++ class needs a distinct name.
  let cppClassName = "StreamTokenType";

  let summary = "The identity and readiness of one SRAM allocation";

  let description = [{
    One token is one runtime sequence point -- a handle plus a descriptor slot --
    and nothing else. It says which allocation a statement works on and what
    that allocation's readiness is; it says nothing about shape.

    Shape belongs to the memref that addresses the same allocation: alloc hands
    out both, views are subviews of that memref, and a statement that has to
    build a descriptor takes the token and the view together. Statements that
    are only about ordering take the token alone, which is all the runtime
    needs: pumem_wait(handle) does not know about sizes.

    A token type has no parameters on purpose. Geometry changes with every view,
    identity does not; keeping the two apart means there is one token type and
    the views stay in memref land, with the whole standard toolkit available.
  }];
}

def PuStream_AllocOp : PUSTREAM_Op<"alloc", [Pure]> {
  let summary = "Allocate an SRAM range: a token and the memref that addresses it";

  let description = [{
    The allocation is the async unit, so it yields the token -- identity plus
    readiness -- together with the memref the synchronous side addresses it by.
    Subviews of that memref are views of the same allocation, and a statement
    pairs the token with whichever view it works on.

    Example:
      %tok, %m = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  }];
  let arguments = (ins);
  let results = (outs AnyType:$token, AnyType:$memref);
  let assemblyFormat = "`(` `)` attr-dict `:` type($token) `,` type($memref)";
  let hasVerifier = 1;
}

def PuStream_TransferOp : PUSTREAM_Op<"transfer", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Enqueue a DRAM <-> SRAM transfer";

  let description = [{
    Exactly one of the two memrefs is the SRAM view, and the direction follows
    from which one it is: SRAM on the source side is a store, SRAM on the
    destination side is a load. The token names the allocation that view belongs
    to and carries its readiness, so the result is that token once the transfer
    is enqueued -- dirty.

    Example:
      %t = pustream.transfer %tok, %dram, %view
        : !pustream.token, memref<256xf32>, memref<256xf32, 128> -> !pustream.token
  }];
  let arguments = (ins AnyType:$token, AnyType:$source, AnyType:$target);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$token `,` $source `,` $target attr-dict `:` type($token) `,` type($source) `,` type($target) `->` type($result)";
  let hasVerifier = 1;
}

def PuStream_CopyOp : PUSTREAM_Op<"copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Enqueue an SRAM to SRAM copy";

  let description = [{
    Both sides are views of SRAM allocations, and each view comes with the token
    of its allocation: the source token is how the verifier knows the source is
    clean, the destination token is what the copy hands back dirty.

    A copy is a statement of its own rather than an outlined dispatch: the copy
    executor is a different queue from the compute executor, and a
    redundant-copy analysis needs to recognise a copy as a copy.

    Example:
      %d = pustream.copy %stok, %src, %dtok, %dst
        : !pustream.token, memref<256xf32, 128>, !pustream.token, memref<256xf32, 128>
          -> !pustream.token
  }];
  let arguments = (ins AnyType:$sourceToken, AnyType:$source, AnyType:$targetToken, AnyType:$target);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$sourceToken `,` $source `,` $targetToken `,` $target attr-dict `:` type($sourceToken) `,` type($source) `,` type($targetToken) `,` type($target) `->` type($result)";
  let hasVerifier = 1;
}

def PuStream_DispatchOp : PUSTREAM_Op<"dispatch", [
  AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Run an outlined executable over a set of allocations";

  let description = [{
    An input is a (token, view) pair or a scalar; an output is always a (token,
    view) pair, because a dispatch writes through it. The pairs come first, in
    order, and the scalars follow -- that is how a fill carries its fill value.

    The read/write split is structural, so no side table of indices is needed.
    The results are the out tokens after the dispatch, dirty, one per pair. A
    dispatch never returns a new view for an out: writing through the allocation
    is what lets several workgroups write disjoint tiles of one buffer, where a
    per-workgroup returned tensor would keep only one of them.

    This is a call. When the callee is an outlined executable, it takes the
    descriptor of each view (and the scalars) in operand order. When it is a
    hand written runtime function, it takes one pointer per view, and its
    template arguments are spelled here: that is how a matmul becomes
    @runtime_matmul<[f32, M, N, K]> instead of a generated loop nest.

    Example:
      %o = pustream.dispatch @body(%itok, %iview)(%otok, %oview)
        : !pustream.token, memref<256xf32, 128>, !pustream.token, memref<256xf32, 128>
          -> !pustream.token
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<ArrayAttr>:$templateArgs,
                       Variadic<AnyType>:$ins,
                       Variadic<AnyType>:$outs);
  let results = (outs Variadic<AnyType>:$results);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PuStream_AwaitOp : PUSTREAM_Op<"await", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait one allocation's sequence point and hand back the clean token";

  let description = [{
    Lowers to pumem_wait(handle) for exactly this allocation: the transfers and
    copies that wrote it. A compute-written token cannot be waited this way --
    the runtime has no per-buffer visibility into the compute queue -- so the
    verifier rejects a dispatch result here and points at await_compute.

    Example: %c = pustream.await %t : !pustream.token -> !pustream.token
  }];
  let arguments = (ins AnyType:$token);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$token attr-dict `:` type($token) `->` type($result)";
  let hasVerifier = 1;
}

def PuStream_AwaitComputeOp : PUSTREAM_Op<"await_compute", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait the compute queue and hand back the clean tokens";

  let description = [{
    Lowers to pumem_wait_dispatch(), which covers everything the compute queue
    has been issued up to the call. It is a barrier, not a per-dispatch wait: the
    runtime has no way to wait for one dispatch, so the operands only name which
    allocations the caller wants back clean.

    Example:
      %o = pustream.await_compute %d : !pustream.token -> !pustream.token
  }];
  let arguments = (ins Variadic<AnyType>:$tokens);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$tokens attr-dict `:` type($tokens) `->` type($results)";
  let hasVerifier = 1;
}

def PuStream_SyncOp : PUSTREAM_Op<"sync", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Drain every queue";

  let description = [{
    Lowers to pumem_sync(). It exists so a kernel can end with one statement
    instead of an await per allocation.

    Example: pustream.sync
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

#endif
