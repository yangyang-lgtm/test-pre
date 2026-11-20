#include "privateuse/Conversion/PustreamToPumem/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Conversion/Common/MemRefParts.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"
#include "privateuse/Utils/Constants.h"

#define GEN_PASS_DEF_PUSTREAMTOPUMEM
#include "privateuse/Conversion/PustreamToPumem/Passes.h.inc"

using namespace mlir;
using namespace mlir::pumem;
using namespace mlir::pustream;

namespace {

bool isSram(MemRefType type)
{
  FailureOr<int64_t> space = getMemRefSpace(type);
  return succeeded(space) && *space == triton::kSramSpace;
}

bool isToken(Type type)
{
  return isa<StreamTokenType>(type);
}

class AllocOpLowering : public OpConversionPattern<pustream::AllocOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pustream::AllocOp op, OneToNOpAdaptor, ConversionPatternRewriter& rewriter) const override
  {
    Location loc    = op.getLoc();
    MemRefType type = cast<MemRefType>(op.getMemref().getType());

    auto memref = memref::AllocOp::create(rewriter, loc, type);
    auto slot   = pumem::DmaSlotOp::create(
      rewriter, loc, DmaHandleType::get(getContext()), DmaMetaType::get(getContext()), memref.getMemref()
    );
    auto handle = pumem::AddressOfOp::create(rewriter, loc, DmaHandlePtrType::get(getContext()), slot.getHandle());
    auto meta   = pumem::AddressOfOp::create(rewriter, loc, DmaMetaPtrType::get(getContext()), slot.getMeta());

    rewriter.replaceOpWithMultiple(
      op, {ValueRange{handle.getPointer(), meta.getPointer()}, ValueRange{memref.getMemref()}}
    );
    return success();
  }
};

class TransferOpLowering : public OpConversionPattern<pustream::TransferOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pustream::TransferOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    Location loc      = op.getLoc();
    auto sourceType   = cast<MemRefType>(op.getSource().getType());
    auto targetType   = cast<MemRefType>(op.getTarget().getType());
    bool sourceIsSram = isSram(sourceType);

    Value sramView  = sourceIsSram ? op.getSource() : op.getTarget();
    Value dramView  = sourceIsSram ? op.getTarget() : op.getSource();
    MemRefType dram = sourceIsSram ? targetType : sourceType;

    FailureOr<int64_t> bytes = getElementByteSize(dram.getElementType(), op);
    if (failed(bytes)) return op.emitError("cannot determine the byte size of the element type");

    ViewDescriptor sramBox = ViewDescriptor::read(rewriter, loc, sramView);
    ViewDescriptor dramBox = ViewDescriptor::read(rewriter, loc, dramView);

    SmallVector<Value> operands{
      adaptor.getToken()[0], adaptor.getToken()[1], sourceIsSram ? dramView : sramView,
      sourceIsSram ? sramView : dramView, constantIndex(rewriter, loc, *bytes)
    };
    operands.append(dramBox.sizes.begin(), dramBox.sizes.end());
    ValueRange dstStrides = sourceIsSram ? ValueRange(dramBox.strides) : ValueRange(sramBox.strides);
    ValueRange srcStrides = sourceIsSram ? ValueRange(sramBox.strides) : ValueRange(dramBox.strides);
    operands.append(dstStrides.begin(), dstStrides.end());
    operands.append(srcStrides.begin(), srcStrides.end());

    auto templateArgs  = rewriter.getArrayAttr({rewriter.getIndexAttr(dram.getRank())});
    const char* callee = sourceIsSram ? "async_copy_sram_to_dram" : "async_copy_dram_to_sram";
    pumem::RuntimeCallOp::create(
      rewriter, loc, TypeRange{}, FlatSymbolRefAttr::get(getContext(), callee), templateArgs, operands
    );

    rewriter.replaceOpWithMultiple(op, {adaptor.getToken()});
    return success();
  }
};

class SramCopyOpLowering : public OpConversionPattern<pustream::CopyOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pustream::CopyOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    Location loc    = op.getLoc();
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    auto targetType = cast<MemRefType>(op.getTarget().getType());

    FailureOr<int64_t> bytes = getElementByteSize(sourceType.getElementType(), op);
    if (failed(bytes)) return op.emitError("cannot determine the byte size of the element type");

    ViewDescriptor source = ViewDescriptor::read(rewriter, loc, op.getSource());
    ViewDescriptor target = ViewDescriptor::read(rewriter, loc, op.getTarget());

    SmallVector<Value> operands{
      adaptor.getTargetToken()[0], adaptor.getTargetToken()[1], op.getTarget(), op.getSource(),
      constantIndex(rewriter, loc, *bytes)
    };
    operands.append(source.sizes.begin(), source.sizes.end());
    operands.append(target.strides.begin(), target.strides.end());
    operands.append(source.strides.begin(), source.strides.end());

    auto templateArgs = rewriter.getArrayAttr({rewriter.getIndexAttr(sourceType.getRank())});
    pumem::RuntimeCallOp::create(
      rewriter, loc, TypeRange{}, FlatSymbolRefAttr::get(getContext(), "async_copy_sram_to_sram"), templateArgs,
      operands
    );
    rewriter.replaceOpWithMultiple(op, {adaptor.getTargetToken()});
    return success();
  }
};

class DispatchOpLowering : public OpConversionPattern<pustream::DispatchOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pustream::DispatchOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    ValueRange ins                     = op.getIns();
    ValueRange outs                    = op.getOuts();
    ArrayRef<ValueRange> convertedIns  = adaptor.getIns();
    ArrayRef<ValueRange> convertedOuts = adaptor.getOuts();

    SmallVector<Value> operands;
    for (unsigned i = 0; i < ins.size(); ++i)
      if (!isToken(ins[i].getType())) operands.push_back(convertedIns[i].front());
    for (unsigned i = 0; i < outs.size(); ++i)
      if (!isToken(outs[i].getType())) operands.push_back(convertedOuts[i].front());

    pumem::RuntimeCallOp::create(
      rewriter, op.getLoc(), TypeRange{}, op.getCalleeAttr(), op.getTemplateArgsAttr(), operands
    );

    SmallVector<ValueRange> results;
    for (unsigned i = 0; i < outs.size(); i += 2) results.push_back(convertedOuts[i]);
    rewriter.replaceOpWithMultiple(op, results);
    return success();
  }
};

class AwaitOpLowering : public OpConversionPattern<pustream::AwaitOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pustream::AwaitOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    pumem::RuntimeCallOp::create(
      rewriter, op.getLoc(), TypeRange{}, FlatSymbolRefAttr::get(getContext(), "pumem_wait"), ArrayAttr{},
      ValueRange{adaptor.getToken()[0]}
    );
    rewriter.replaceOpWithMultiple(op, {adaptor.getToken()});
    return success();
  }
};

class AwaitComputeOpLowering : public OpConversionPattern<pustream::AwaitComputeOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    pustream::AwaitComputeOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    pumem::RuntimeCallOp::create(
      rewriter, op.getLoc(), TypeRange{}, FlatSymbolRefAttr::get(getContext(), "pumem_wait_dispatch"), ArrayAttr{},
      ValueRange{}
    );
    SmallVector<ValueRange> results;
    for (ValueRange token : adaptor.getTokens()) results.push_back(token);
    rewriter.replaceOpWithMultiple(op, results);
    return success();
  }
};

class SyncOpLowering : public OpConversionPattern<pustream::SyncOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pustream::SyncOp op, OneToNOpAdaptor, ConversionPatternRewriter& rewriter) const override
  {
    pumem::RuntimeCallOp::create(
      rewriter, op.getLoc(), TypeRange{}, FlatSymbolRefAttr::get(getContext(), "pumem_sync"), ArrayAttr{}, ValueRange{}
    );
    rewriter.eraseOp(op);
    return success();
  }
};

struct PustreamToPumemPass : public ::impl::PustreamToPumemBase<PustreamToPumemPass>
{
  using PustreamToPumemBase::PustreamToPumemBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<memref::MemRefDialect, pumem::PUMemDialect, func::FuncDialect, arith::ArithDialect>();
  }

  void runOnOperation() override
  {
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](StreamTokenType type, SmallVectorImpl<Type>& results) {
      results.push_back(DmaHandlePtrType::get(type.getContext()));
      results.push_back(DmaMetaPtrType::get(type.getContext()));
      return success();
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<PuStreamDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    RewritePatternSet patterns(&getContext());

    // clang-format off
    patterns.add<
      AllocOpLowering,
      TransferOpLowering,
      SramCopyOpLowering,
      DispatchOpLowering,
      AwaitOpLowering,
      AwaitComputeOpLowering,
      SyncOpLowering
    >(converter, &getContext());
    // clang-format on

    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createPustreamToPumemPass()
{
  return std::make_unique<PustreamToPumemPass>();
}
