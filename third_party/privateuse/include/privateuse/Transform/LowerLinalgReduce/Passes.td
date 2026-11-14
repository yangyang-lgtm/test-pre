#ifndef PRIVATEUSE_TRANSFORM_LOWERLINALGREDUCE_PASSES_TD
#define PRIVATEUSE_TRANSFORM_LOWERLINALGREDUCE_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerLinalgReduce : Pass<"privateuse-lower-linalg-reduce", "mlir::ModuleOp"> {
  let summary = "Lower single-axis reduction linalg.generic ops to scf loops";
  let description = [{
    Rewrites every buffer-level linalg.generic with exactly one reduction
    iterator into a nest of scf.for loops over the parallel dimensions and an
    inner scf.for over the reduction dimension. The accumulator starts from
    the first input element, matching Triton reducers whose combine function
    is associative and never depends on the (uninitialized) init buffer.

    Ops that do not match (dynamic shapes, several reduction iterators, fused
    producer prefixes, ...) are left in place for the generic
    linalg-to-loops lowering.
  }];
  let constructor = "mlir::triton::createLowerLinalgReducePass()";
}

#endif
