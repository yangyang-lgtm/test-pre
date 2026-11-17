include "mlir/Pass/PassBase.td"

def AnalyzeDataflow : Pass<"privateuse-analyze-dataflow", "mlir::ModuleOp"> {
  let summary = "Report the static DRAM<->SRAM dataflow model";
  let description = [{
    Builds the dataflow model over the bufferized privateuse IR and prints one
    report line per loop, place, transfer, carrier, station and access. The pass
    never mutates the IR.
  }];
  let constructor = "mlir::triton::createAnalyzeDataflowPass()";
}