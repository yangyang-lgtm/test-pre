#ifndef PRIVATEUSE_CONVERSION_TOPUC_PASSES_TD
#define PRIVATEUSE_CONVERSION_TOPUC_PASSES_TD

include "mlir/Pass/PassBase.td"

def ToPuc : Pass<"privateuse-to-puc", "mlir::ModuleOp"> {
  let summary = "Lower the whole program into the puc emission dialect";
  let description = [{
    One conversion for the function layer and every payload dialect: the puc
    dialect keeps only the operations the C emitter prints directly (typed
    pointers, constants, pointer arithmetic, scalar load/store, functions,
    calls, puc.for / puc.if / puc.yield, and the kernel launcher), everything
    else becomes a `puc.call` into the runtime with a bare callee name.

    Lowered in place: pumem.addptr / pumem.load / pumem.store /
    pumem.point_cast become their puc counterparts, pumem.for / pumem.if /
    pumem.yield become puc.for / puc.if / puc.yield, puarith.constant becomes
    puc.constant. `func.func` / `func.return` / `func.call` become
    puc.func / puc.return / puc.call.

    Lowered to runtime calls: pumem.get_physical_id (@get_physical_id),
    pumem.alloc (@alloc), pumem.dealloc (@dealloc), pumem.copy (@copy),
    pumem.gather (@gather), pumem.scatter (@scatter) and
    pumem.runtime_call (its own callee and template arguments). The
    puarith / pumath value ops go through one table-driven pattern and become
    `scalar_<mnemonic>` calls, with `_<src>_<dst>` appended for casts and
    `_<predicate>` for comparisons. Scalar index casts keep the sentinel
    callee `/*index cast*/`. The flat operand layout of every call is the
    operand order of the op that produced it; the runtime splits variadic
    operand groups with the template arguments carried by the call.
  }];
  let dependentDialects = [
    "::mlir::pumem::PUMemDialect",
    "::mlir::puarith::PuArithDialect",
    "::mlir::pumath::PuMathDialect",
    "::mlir::puc::PucDialect",
    "::mlir::func::FuncDialect"
  ];
  let constructor = "mlir::triton::createToPucPass()";
}

#endif
