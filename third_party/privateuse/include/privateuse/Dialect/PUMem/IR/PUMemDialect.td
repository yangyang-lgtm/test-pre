#ifndef PUMEM_DIALECT_TD
#define PUMEM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PUMem_Dialect : Dialect {
  let name = "pumem";

  let cppNamespace = "::mlir::pumem";

  let summary = "PUM memory dialect";
}

class PUMEM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PUMem_Dialect, mnemonic, traits> {
}

def PUMem_GetPhysicalIdOp : PUMEM_Op<"get_physical_id", [Pure]> {
  let summary = "Get the id of the current physical thread";

  let description = [{
    Returns the linear physical thread id set by the launch loop, in
    [0, physical_threads_num). The program loop starts from it and strides by
    the number of physical threads to cover the whole grid.

    Example: %tid = pumem.get_physical_id : index
  }];
  let results = (outs Index:$id);
  let assemblyFormat = "attr-dict `:` type($id)";
}

def PUMem_GatherOp : PUMEM_Op<"gather"> {
  let summary = "Gather elements from a flat buffer into a tile";

  let description = [{
    Writes one element into the destination tile for every element of the
    offset buffer, both shaped alike:

      destination[i] = source[indices[i]]

    `source` is a flat runtime buffer addressed in elements; `indices` is a
    runtime buffer of element offsets. The op has no results and mutates the
    destination tile in place, like memref.copy.
  }];
  let arguments = (ins Arg<AnyMemRef, "flat buffer read at the given element offsets", [MemRead]>:$source,
                       Arg<AnyMemRef, "per-element offsets", [MemRead]>:$indices,
                       Arg<AnyMemRef, "tile written with the gathered values", [MemWrite]>:$destination);
  let assemblyFormat = "$source `,` $indices `,` $destination attr-dict `:` type($source) `,` type($indices) `,` type($destination)";
  let hasVerifier = 1;
}

def PUMem_ScatterOp : PUMEM_Op<"scatter"> {
  let summary = "Scatter elements from a tile into a flat buffer";

  let description = [{
    Reads one element from the source tile for every element of the offset
    buffer, both shaped alike, and writes it into the flat destination buffer:

      destination[indices[i]] = source[i]

    `destination` is a flat runtime buffer addressed in elements; `indices` is
    a runtime buffer of element offsets. The op has no results and mutates the
    destination buffer in place, like memref.copy.
  }];
  let arguments = (ins Arg<AnyMemRef, "tile read in order", [MemRead]>:$source,
                       Arg<AnyMemRef, "per-element offsets", [MemRead]>:$indices,
                       Arg<AnyMemRef, "flat buffer written at the given element offsets", [MemWrite]>:$destination);
  let assemblyFormat = "$source `,` $indices `,` $destination attr-dict `:` type($source) `,` type($indices) `,` type($destination)";
  let hasVerifier = 1;
}

#endif
