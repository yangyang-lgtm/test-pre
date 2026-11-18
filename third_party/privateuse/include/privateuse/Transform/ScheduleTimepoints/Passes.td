#ifndef PRIVATEUSE_TRANSFORM_SCHEDULETIMEPOINTS_PASSES_TD
#define PRIVATEUSE_TRANSFORM_SCHEDULETIMEPOINTS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ScheduleTimepoints : Pass<"privateuse-schedule-timepoints", "mlir::ModuleOp"> {
  let summary = "Wire async timepoints into awaits and region drains";
  let description = [{
    Makes the async stream IR well-ordered. Within a block every async
    operation waits for earlier writes (WAW/RAW) and reads (WAR) of the
    buffers it touches: async consumers receive the producer timepoints
    through their `awaits` operand, synchronous consumers get an explicit
    `pustream.await`. Every region (scf.if / scf.for) drains the timepoints it
    produced before it completes, so writes issued inside a region are
    complete once control leaves it. Timepoints still pending at the function
    exit are awaited before the return.

    Buffers are identified by their base memref after stripping view-like ops,
    so a copy into a subview hazards against the whole allocation.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::pustream::PuStreamDialect"
  ];
  let constructor = "mlir::triton::createScheduleTimepointsPass()";
}

#endif
