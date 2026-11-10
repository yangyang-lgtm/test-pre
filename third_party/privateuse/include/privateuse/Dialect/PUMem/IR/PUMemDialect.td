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

#endif
