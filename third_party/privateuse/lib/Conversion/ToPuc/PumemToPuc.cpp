#include "privateuse/Conversion/ToPuc/PucPatterns.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

using namespace mlir;

namespace {

puc::CallOp createRuntimeCall(
  OpBuilder& builder, Location loc, TypeRange results, StringRef callee, ValueRange operands,
  ArrayRef<std::string> templateArgs = {}
)
{
  ArrayAttr templates;
  if (!templateArgs.empty()) {
    SmallVector<Attribute> attrs;
    for (const std::string& arg : templateArgs) attrs.push_back(builder.getStringAttr(arg));
    templates = builder.getArrayAttr(attrs);
  }
  return puc::CallOp::create(
    builder, loc, results, FlatSymbolRefAttr::get(builder.getContext(), callee), templates, operands
  );
}

std::string cTypeName(Type type)
{
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 1 ? "int8_t" : "int" + std::to_string(intTy.getWidth()) + "_t";
  if (isa<IndexType>(type)) return "int64_t";
  if (isa<Float32Type>(type)) return "float";
  if (isa<Float64Type>(type)) return "double";
  if (isa<Float16Type>(type)) return "_Float16";
  if (isa<BFloat16Type>(type)) return "bfloat16";
  return "";
}

std::string templateArgString(Attribute attr)
{
  if (auto typeAttr = dyn_cast<TypeAttr>(attr)) return cTypeName(typeAttr.getValue());
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) return std::to_string(intAttr.getInt());
  return "";
}

struct GetPhysicalIdOpConversion : OpConversionPattern<pumem::GetPhysicalIdOp>
{
  using OpConversionPattern<pumem::GetPhysicalIdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::GetPhysicalIdOp op, pumem::GetPhysicalIdOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    rewriter.replaceOp(op, createRuntimeCall(rewriter, op.getLoc(), {op.getType()}, "get_physical_id", {}));
    return success();
  }
};

struct AllocOpConversion : OpConversionPattern<pumem::AllocOp>
{
  using OpConversionPattern<pumem::AllocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pumem::AllocOp op, pumem::AllocOp::Adaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    auto call = createRuntimeCall(
      rewriter, op.getLoc(), {op.getPtr().getType()}, "alloc", {adaptor.getBytes(), adaptor.getAlign()}
    );
    rewriter.replaceOp(op, call->getResults());
    return success();
  }
};

struct DeallocOpConversion : OpConversionPattern<pumem::DeallocOp>
{
  using OpConversionPattern<pumem::DeallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::DeallocOp op, pumem::DeallocOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    rewriter.replaceOp(op, createRuntimeCall(rewriter, op.getLoc(), {}, "dealloc", {adaptor.getPtr()}));
    return success();
  }
};

struct AddPtrOpConversion : OpConversionPattern<pumem::AddPtrOp>
{
  using OpConversionPattern<pumem::AddPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::AddPtrOp op, pumem::AddPtrOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    rewriter.replaceOpWithNewOp<puc::AddPtrOp>(op, op.getResult().getType(), adaptor.getPtr(), adaptor.getOffset());
    return success();
  }
};

struct PointCastOpConversion : OpConversionPattern<pumem::PointCastOp>
{
  using OpConversionPattern<pumem::PointCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::PointCastOp op, pumem::PointCastOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    rewriter.replaceOpWithNewOp<puc::PointCastOp>(op, op.getResult().getType(), adaptor.getPtr());
    return success();
  }
};

struct LoadOpConversion : OpConversionPattern<pumem::LoadOp>
{
  using OpConversionPattern<pumem::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pumem::LoadOp op, pumem::LoadOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<puc::LoadOp>(op, op.getValue().getType(), adaptor.getPtr());
    return success();
  }
};

struct StoreOpConversion : OpConversionPattern<pumem::StoreOp>
{
  using OpConversionPattern<pumem::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pumem::StoreOp op, pumem::StoreOp::Adaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    rewriter.replaceOpWithNewOp<puc::StoreOp>(op, adaptor.getValue(), adaptor.getPtr());
    return success();
  }
};

struct DmaSlotOpConversion : OpConversionPattern<pumem::DmaSlotOp>
{
  using OpConversionPattern<pumem::DmaSlotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::DmaSlotOp op, pumem::DmaSlotOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    auto function = op->getParentOfType<puc::FuncOp>();
    if (!function) return op.emitError("a slot needs an enclosing function");

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&function.getBody().front());
    auto handle = puc::LocalOp::create(rewriter, op.getLoc(), op.getHandle().getType(), Value{});
    auto meta   = puc::LocalOp::create(rewriter, op.getLoc(), op.getMeta().getType(), Value{});
    rewriter.replaceOp(op, {handle.getResult(), meta.getResult()});
    return success();
  }
};

struct AddressOfOpConversion : OpConversionPattern<pumem::AddressOfOp>
{
  using OpConversionPattern<pumem::AddressOfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::AddressOfOp op, pumem::AddressOfOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    auto pointer = puc::AddressOfOp::create(rewriter, op.getLoc(), op.getPointer().getType(), adaptor.getValue());
    rewriter.replaceOp(op, pointer.getPointer());
    return success();
  }
};

struct CopyOpConversion : OpConversionPattern<pumem::CopyOp>
{
  using OpConversionPattern<pumem::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pumem::CopyOp op, pumem::CopyOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Value> operands = {adaptor.getDst(), adaptor.getSrc(), adaptor.getByteSize()};
    operands.append(adaptor.getSizes().begin(), adaptor.getSizes().end());
    operands.append(adaptor.getDstStrides().begin(), adaptor.getDstStrides().end());
    operands.append(adaptor.getSrcStrides().begin(), adaptor.getSrcStrides().end());
    std::string rank   = std::to_string(adaptor.getSizes().size());
    std::string callee = "copy_" + op.getDir().str();
    rewriter.replaceOp(op, createRuntimeCall(rewriter, op.getLoc(), {}, callee, operands, {rank}));
    return success();
  }
};

struct GatherOpConversion : OpConversionPattern<pumem::GatherOp>
{
  using OpConversionPattern<pumem::GatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::GatherOp op, pumem::GatherOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    SmallVector<Value> operands = {
      adaptor.getSource(), adaptor.getSourceStride(), adaptor.getIndices(), adaptor.getDestination()
    };
    operands.append(adaptor.getSizes().begin(), adaptor.getSizes().end());
    operands.append(adaptor.getIndicesStrides().begin(), adaptor.getIndicesStrides().end());
    operands.append(adaptor.getDestinationStrides().begin(), adaptor.getDestinationStrides().end());
    bool hasMask  = adaptor.getMask() != nullptr;
    bool hasOther = adaptor.getOther() != nullptr;
    if (hasMask) {
      operands.push_back(adaptor.getMask());
      operands.append(adaptor.getMaskStrides().begin(), adaptor.getMaskStrides().end());
    }
    if (hasOther) operands.push_back(adaptor.getOther());

    std::vector<std::string> templateArgs = {
      std::to_string(adaptor.getSizes().size()), hasMask ? "true" : "false", hasOther ? "true" : "false"
    };
    rewriter.replaceOp(op, createRuntimeCall(rewriter, op.getLoc(), {}, "gather", operands, templateArgs));
    return success();
  }
};

struct ScatterOpConversion : OpConversionPattern<pumem::ScatterOp>
{
  using OpConversionPattern<pumem::ScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::ScatterOp op, pumem::ScatterOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    SmallVector<Value> operands = {
      adaptor.getSource(), adaptor.getIndices(), adaptor.getDestination(), adaptor.getDestinationStride()
    };
    operands.append(adaptor.getSizes().begin(), adaptor.getSizes().end());
    operands.append(adaptor.getSourceStrides().begin(), adaptor.getSourceStrides().end());
    operands.append(adaptor.getIndicesStrides().begin(), adaptor.getIndicesStrides().end());
    bool hasMask = adaptor.getMask() != nullptr;
    if (hasMask) {
      operands.push_back(adaptor.getMask());
      operands.append(adaptor.getMaskStrides().begin(), adaptor.getMaskStrides().end());
    }

    std::vector<std::string> templateArgs = {std::to_string(adaptor.getSizes().size()), hasMask ? "true" : "false"};
    rewriter.replaceOp(op, createRuntimeCall(rewriter, op.getLoc(), {}, "scatter", operands, templateArgs));
    return success();
  }
};

struct RuntimeCallOpConversion : OpConversionPattern<pumem::RuntimeCallOp>
{
  using OpConversionPattern<pumem::RuntimeCallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pumem::RuntimeCallOp op, pumem::RuntimeCallOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    std::vector<std::string> templateArgs;
    if (auto templates = op.getTemplateArgs()) {
      for (Attribute attr : *templates) {
        std::string arg = templateArgString(attr);
        if (arg.empty()) return op.emitOpError("unsupported template argument ") << attr;
        templateArgs.push_back(std::move(arg));
      }
    }
    auto call = createRuntimeCall(
      rewriter, op.getLoc(), op.getResultTypes(), op.getCallee(), adaptor.getOperands(), templateArgs
    );
    rewriter.replaceOp(op, call->getResults());
    return success();
  }
};

struct ForOpConversion : OpConversionPattern<pumem::ForOp>
{
  using OpConversionPattern<pumem::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pumem::ForOp op, pumem::ForOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto newOp = puc::ForOp::create(
      rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep(), adaptor.getInitArgs(),
      nullptr
    );
    Region& body = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), body, body.end());
    rewriter.eraseBlock(&body.front());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct IfOpConversion : OpConversionPattern<pumem::IfOp>
{
  using OpConversionPattern<pumem::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pumem::IfOp op, pumem::IfOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    bool hasElse = !op.getElseRegion().empty();
    auto newOp   = puc::IfOp::create(
      rewriter, op.getLoc(), op.getResultTypes(), adaptor.getCondition(),
      /*addThenBlock=*/true, /*addElseBlock=*/hasElse
    );
    Region& thenRegion = newOp.getThenRegion();
    rewriter.inlineRegionBefore(op.getThenRegion(), thenRegion, thenRegion.end());
    rewriter.eraseBlock(&thenRegion.front());
    if (hasElse) {
      Region& elseRegion = newOp.getElseRegion();
      rewriter.inlineRegionBefore(op.getElseRegion(), elseRegion, elseRegion.end());
      rewriter.eraseBlock(&elseRegion.front());
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct YieldOpConversion : OpConversionPattern<pumem::YieldOp>
{
  using OpConversionPattern<pumem::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pumem::YieldOp op, pumem::YieldOp::Adaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    rewriter.replaceOpWithNewOp<puc::YieldOp>(op, adaptor.getResults());
    return success();
  }
};

} // namespace

void mlir::triton::populatePumemToPucPatterns(RewritePatternSet& patterns, TypeConverter& typeConverter)
{
  // clang-format off
  patterns.add<
    GetPhysicalIdOpConversion,
    AllocOpConversion,
    DeallocOpConversion,
    AddPtrOpConversion,
    PointCastOpConversion,
    LoadOpConversion,
    StoreOpConversion,
    DmaSlotOpConversion,
    AddressOfOpConversion,
    CopyOpConversion,
    GatherOpConversion,
    ScatterOpConversion,
    RuntimeCallOpConversion,
    ForOpConversion,
    IfOpConversion,
    YieldOpConversion
  >(typeConverter, patterns.getContext());
  // clang-format on
}
