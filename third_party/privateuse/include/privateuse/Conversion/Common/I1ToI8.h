#ifndef TRITON_CONVERSION_COMMON_I1TOI8_H
#define TRITON_CONVERSION_COMMON_I1TOI8_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

inline Type i1ToI8(Type type)
{
  auto intTy = dyn_cast<IntegerType>(type);
  if (!intTy || intTy.getWidth() != 1 || !intTy.isSignless()) return type;
  return IntegerType::get(type.getContext(), 8);
}

inline Type promoteI1Element(Type type)
{
  if (auto vecTy = dyn_cast<VectorType>(type))
    return VectorType::get(vecTy.getShape(), i1ToI8(vecTy.getElementType()), vecTy.getScalableDims());
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensorTy.getShape(), i1ToI8(tensorTy.getElementType()), tensorTy.getEncoding());
  if (auto tensorTy = dyn_cast<UnrankedTensorType>(type))
    return UnrankedTensorType::get(i1ToI8(tensorTy.getElementType()));
  if (auto memrefTy = dyn_cast<MemRefType>(type))
    return MemRefType::get(
      memrefTy.getShape(), i1ToI8(memrefTy.getElementType()), memrefTy.getLayout(), memrefTy.getMemorySpace()
    );
  if (auto memrefTy = dyn_cast<UnrankedMemRefType>(type))
    return UnrankedMemRefType::get(i1ToI8(memrefTy.getElementType()), memrefTy.getMemorySpace());
  return i1ToI8(type);
}

inline void addI1ToI8Conversions(TypeConverter& typeConverter)
{
  typeConverter.addConversion([](Type type) -> std::optional<Type> { return type; });
  typeConverter.addConversion([](IntegerType type) -> std::optional<Type> { return i1ToI8(type); });
  typeConverter.addConversion([](VectorType type) -> std::optional<Type> { return promoteI1Element(type); });
  typeConverter.addConversion([](RankedTensorType type) -> std::optional<Type> { return promoteI1Element(type); });
  typeConverter.addConversion([](UnrankedTensorType type) -> std::optional<Type> { return promoteI1Element(type); });
  typeConverter.addConversion([](MemRefType type) -> std::optional<Type> { return promoteI1Element(type); });
  typeConverter.addConversion([](UnrankedMemRefType type) -> std::optional<Type> { return promoteI1Element(type); });

  auto materialize = [](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs).getResult(0);
  };
  typeConverter.addTargetMaterialization(materialize);
  typeConverter.addSourceMaterialization(materialize);
}

template <typename OpT> struct I1OpConversion : public OpConversionPattern<OpT>
{
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) return failure();

    OperationState state(op.getLoc(), op->getName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

inline void populateI1ToI8Patterns(RewritePatternSet& patterns, TypeConverter& typeConverter)
{
  // clang-format off
  patterns.add<
    I1OpConversion<memref::AllocOp>,
    I1OpConversion<memref::AllocaOp>,
    I1OpConversion<memref::CastOp>,
    I1OpConversion<memref::CopyOp>,
    I1OpConversion<memref::DeallocOp>,
    I1OpConversion<memref::LoadOp>,
    I1OpConversion<memref::StoreOp>,
    I1OpConversion<memref::ReinterpretCastOp>,
    I1OpConversion<memref::SubViewOp>,
    I1OpConversion<memref::CollapseShapeOp>,
    I1OpConversion<memref::ExpandShapeOp>,
    I1OpConversion<vector::TransferReadOp>,
    I1OpConversion<vector::TransferWriteOp>
  >(typeConverter, patterns.getContext());
  // clang-format on

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

inline void markI1ToI8OpsDynamicallyLegal(ConversionTarget& target, TypeConverter& typeConverter)
{
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) && typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>([&](Operation* op) { return typeConverter.isLegal(op); });

  auto legal = [&](Operation* op) { return typeConverter.isLegal(op); };
  // clang-format off
  target.addDynamicallyLegalOp<
    memref::AllocOp,
    memref::AllocaOp,
    memref::CastOp,
    memref::CopyOp,
    memref::DeallocOp,
    memref::LoadOp,
    memref::StoreOp,
    memref::ReinterpretCastOp,
    memref::SubViewOp,
    memref::CollapseShapeOp,
    memref::ExpandShapeOp,
    vector::TransferReadOp,
    vector::TransferWriteOp
  >(legal);
  // clang-format on
}

}
}

#endif
