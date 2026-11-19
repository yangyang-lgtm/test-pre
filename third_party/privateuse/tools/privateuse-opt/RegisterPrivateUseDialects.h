#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "privateuse/Conversion/AccessLowering/Passes.h"
#include "privateuse/Conversion/ArithToLinalg/Passes.h"
#include "privateuse/Conversion/ArithToPuarith/Passes.h"
#include "privateuse/Conversion/Bufferize/Passes.h"
#include "privateuse/Conversion/MathToPumath/Passes.h"
#include "privateuse/Conversion/MemrefToPumem/Passes.h"
#include "privateuse/Conversion/ScfToPumem/Passes.h"
#include "privateuse/Conversion/ToPuc/Passes.h"
#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h"
#include "privateuse/Dialect/PuMath/IR/PuMathDialect.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"
#include "privateuse/Transform/LowerLinalgReduce/Passes.h"
#include "privateuse/Transform/PidLoop/Passes.h"
#include "privateuse/Transform/PucLaunchKernel/Passes.h"
#include "privateuse/Transform/PucLocalizeCarried/Passes.h"
#include "privateuse/Transform/PucSimplifyAssign/Passes.h"
#include "privateuse/Transform/PucVoidPtrArgs/Passes.h"
#include "privateuse/Transform/SinkConstantCallArgs/Passes.h"
#include "privateuse/Transform/SpecializeKernel/Passes.h"
#include "privateuse/Utils/AccessAnalysis/Passes.h"

#include "mlir/InitAllPasses.h"

inline void registerPrivateUseDialects(mlir::DialectRegistry& registry)
{
  mlir::registerAllPasses();
  mlir::registerLinalgPasses();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::triton::registerTritonPasses();
  mlir::triton::registerAccessAnalysisPasses();
  mlir::triton::registerAccessLoweringPasses();
  mlir::triton::registerArithToLinalgPasses();
  mlir::triton::registerArithToPuarithPass();
  mlir::triton::registerBufferizePasses();
  mlir::triton::registerMathToPumathPass();
  mlir::triton::registerMemrefToPumemPasses();
  mlir::triton::registerScfToPumemPass();
  mlir::triton::registerToPucPasses();
  mlir::triton::registerLowerLinalgReducePasses();
  mlir::triton::registerPidLoopPasses();
  mlir::triton::registerPucLaunchKernelPasses();
  mlir::triton::registerPucLocalizeCarriedPass();
  mlir::triton::registerPucSimplifyAssignPass();
  mlir::triton::registerPucVoidPtrArgsPass();
  mlir::triton::registerSinkConstantCallArgsPasses();
  mlir::triton::registerSpecializeKernelPass();
  // clang-format off
  registry.insert<
      mlir::pumem::PUMemDialect,
      mlir::puarith::PuArithDialect,
      mlir::pumath::PuMathDialect,
      mlir::puc::PucDialect,
      mlir::ptr::PtrDialect,
      mlir::affine::AffineDialect,
      mlir::triton::TritonDialect,
      mlir::cf::ControlFlowDialect,
      mlir::math::MathDialect,
      mlir::arith::ArithDialect,
      mlir::scf::SCFDialect,
      mlir::linalg::LinalgDialect,
      mlir::func::FuncDialect,
      mlir::vector::VectorDialect,
      mlir::tensor::TensorDialect,
      mlir::memref::MemRefDialect,
      mlir::bufferization::BufferizationDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::instrument::TritonInstrumentDialect,
      mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,
      mlir::NVVM::NVVMDialect,
      mlir::triton::nvws::NVWSDialect,
      mlir::ROCDL::ROCDLDialect,
      mlir::triton::gluon::GluonDialect
  >();
  // clang-format on
}
