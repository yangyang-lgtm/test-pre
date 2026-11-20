#include "privateuse/Transform/MemrefToPuStream/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"
#include "privateuse/Utils/Constants.h"

#include <algorithm>
#include <optional>
#include <string>

#define GEN_PASS_DECL_MEMREFTOPUSTREAM
#include "privateuse/Transform/MemrefToPuStream/Passes.h.inc"

#define GEN_PASS_DEF_MEMREFTOPUSTREAM
#include "privateuse/Transform/MemrefToPuStream/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isSram(Type type)
{
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref) return false;
  Attribute space = memref.getMemorySpace();
  if (!space) return false;
  auto integer = dyn_cast<IntegerAttr>(space);
  return integer && integer.getInt() == kSramSpace;
}

bool isToken(Type type)
{
  return isa<pustream::StreamTokenType>(type);
}

pustream::AllocOp allocationOfView(Value view)
{
  while (auto viewLike = view.getDefiningOp<ViewLikeOpInterface>()) view = viewLike.getViewSource();
  return view.getDefiningOp<pustream::AllocOp>();
}

pustream::AllocOp allocationOfToken(Value token)
{
  while (true) {
    Operation* def = token.getDefiningOp();
    if (!def) return {};
    if (auto alloc = dyn_cast<pustream::AllocOp>(def)) return alloc;
    if (auto await = dyn_cast<pustream::AwaitOp>(def)) {
      token = await.getToken();
      continue;
    }
    if (auto awaitCompute = dyn_cast<pustream::AwaitComputeOp>(def)) {
      token = awaitCompute.getTokens()[cast<OpResult>(token).getResultNumber()];
      continue;
    }
    if (auto transfer = dyn_cast<pustream::TransferOp>(def)) {
      token = transfer.getToken();
      continue;
    }
    if (auto copy = dyn_cast<pustream::CopyOp>(def)) {
      token = copy.getTargetToken();
      continue;
    }
    if (auto dispatch = dyn_cast<pustream::DispatchOp>(def)) {
      ValueRange outs = dispatch.getOuts();
      unsigned index  = cast<OpResult>(token).getResultNumber();
      if (outs.size() != 2 * dispatch.getResults().size()) return {};
      token = outs[2 * index];
      continue;
    }
    return {};
  }
}

FailureOr<Value> tokenBefore(pustream::AllocOp allocation, Operation* use)
{
  Operation* from = use;
  for (Block* block = use->getBlock(); block;) {
    for (Operation* op = from->getPrevNode(); op; op = op->getPrevNode()) {
      for (Value result : op->getResults()) {
        if (!isToken(result.getType()) || allocationOfToken(result) != allocation) continue;
        if (isa<pustream::TransferOp, pustream::CopyOp, pustream::DispatchOp>(op)) {
          op->emitError("this allocation is still in flight; await it before using it again");
          return failure();
        }
        return result;
      }
    }
    Operation* parent = block->getParentOp();
    if (!parent) break;
    from  = parent;
    block = parent->getBlock();
  }
  return failure();
}

LogicalResult appendPair(Operation* at, Value view, SmallVectorImpl<Value>& pairs)
{
  auto alloc = allocationOfView(view);
  if (!alloc) return failure();
  auto token = tokenBefore(alloc, at);
  if (failed(token)) return failure();
  pairs.push_back(*token);
  pairs.push_back(view);
  return success();
}

bool classify(linalg::LinalgOp op, SmallVectorImpl<Value>& ins, SmallVectorImpl<Value>& outs)
{
  for (OpOperand& operand : op->getOpOperands()) {
    Value value = operand.get();
    Type type   = value.getType();
    if (op.isDpsInit(&operand)) {
      if (!isa<MemRefType>(type) || !allocationOfView(value)) return false;
      if (!llvm::is_contained(outs, value)) outs.push_back(value);
      continue;
    }
    if (isa<MemRefType>(type) && !allocationOfView(value)) return false;
    if (!isa<MemRefType>(type) && !type.isIntOrIndexOrFloat()) return false;
    if (!llvm::is_contained(ins, value)) ins.push_back(value);
  }
  for (Value value : ins)
    if (llvm::is_contained(outs, value)) return false;
  return !outs.empty();
}

bool isComputable(linalg::LinalgOp op)
{
  if (!op.hasPureBufferSemantics() || op->getNumResults() != 0) return false;
  if (op->getParentOfType<pustream::DispatchOp>()) return false;
  if (auto function = op->getParentOfType<func::FuncOp>())
    if (function->hasAttr(kOutlinedAttr)) return false;
  SmallVector<Value> ins, outs;
  return classify(op, ins, outs);
}

struct MatmulShape
{
  Type element;
  int64_t M, N, K;
};

std::optional<MatmulShape> matmulShape(linalg::MatmulOp op)
{
  auto a = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto b = dyn_cast<MemRefType>(op.getInputs()[1].getType());
  auto c = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  if (!a || !b || !c) return std::nullopt;
  if (!a.hasStaticShape() || !b.hasStaticShape() || !c.hasStaticShape()) return std::nullopt;
  if (a.getRank() != 2 || b.getRank() != 2 || c.getRank() != 2) return std::nullopt;
  if (b.getShape()[0] != a.getShape()[1]) return std::nullopt;
  if (c.getShape() != ArrayRef<int64_t>{a.getShape()[0], b.getShape()[1]}) return std::nullopt;
  if (a.getElementType() != b.getElementType() || a.getElementType() != c.getElementType()) return std::nullopt;
  return MatmulShape{a.getElementType(), a.getShape()[0], b.getShape()[1], a.getShape()[1]};
}

struct TransposeShape
{
  Type element;
  int64_t rows, cols;
};

std::optional<TransposeShape> transposeShape(linalg::TransposeOp op)
{
  auto src = dyn_cast<MemRefType>(op.getInput().getType());
  auto dst = dyn_cast<MemRefType>(op.getInit().getType());
  if (!src || !dst) return std::nullopt;
  if (!src.hasStaticShape() || !dst.hasStaticShape()) return std::nullopt;
  if (src.getRank() != 2 || dst.getRank() != 2) return std::nullopt;
  if (src.getElementType() != dst.getElementType()) return std::nullopt;
  if (dst.getShape() != ArrayRef<int64_t>{src.getShape()[1], src.getShape()[0]}) return std::nullopt;
  if (op.getPermutation() != ArrayRef<int64_t>{1, 0}) return std::nullopt;
  return TransposeShape{src.getElementType(), src.getShape()[0], src.getShape()[1]};
}

void emitDispatch(
  PatternRewriter& rewriter, Location loc, FlatSymbolRefAttr callee, ArrayAttr templateArgs, ValueRange ins,
  ValueRange outs
)
{
  OperationState state(loc, pustream::DispatchOp::getOperationName());
  state.addOperands(ins);
  state.addOperands(outs);
  state.addAttribute("callee", callee);
  if (templateArgs) state.addAttribute("templateArgs", templateArgs);
  state.addAttribute(
    "operandSegmentSizes",
    rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(ins.size()), static_cast<int32_t>(outs.size())})
  );

  for (unsigned i = 0; i < outs.size(); i += 2) state.addTypes(outs[i].getType());
  auto dispatch = cast<pustream::DispatchOp>(rewriter.create(state));
  SmallVector<Type> results(dispatch.getResults().getTypes());
  pustream::AwaitComputeOp::create(rewriter, loc, results, dispatch.getResults());
}

class AllocOpLowering : public OpRewritePattern<memref::AllocOp>
{
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op, PatternRewriter& rewriter) const override
  {
    if (!isSram(op.getType())) return op.emitError("expected an SRAM allocation");

    auto token = pustream::StreamTokenType::get(rewriter.getContext());
    auto alloc = pustream::AllocOp::create(rewriter, op.getLoc(), token, op.getType());
    rewriter.replaceOp(op, alloc.getMemref());
    return success();
  }
};

class CopyOpLowering : public OpRewritePattern<memref::CopyOp>
{
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy, PatternRewriter& rewriter) const override
  {
    bool sourceSram = isSram(copy.getSource().getType());
    bool targetSram = isSram(copy.getTarget().getType());
    if (!sourceSram && !targetSram)
      return copy.emitError("expected a copy that touches SRAM; a DRAM to DRAM copy is not a statement");

    Location loc = copy.getLoc();

    if (sourceSram && targetSram) {
      auto sourceAlloc = allocationOfView(copy.getSource());
      auto targetAlloc = allocationOfView(copy.getTarget());
      if (!sourceAlloc || !targetAlloc)
        return copy.emitError("cannot express this copy: an SRAM side does not trace back to an allocation");
      if (sourceAlloc == targetAlloc) return copy.emitError("copying an allocation onto itself is not a statement");

      auto sourceToken = tokenBefore(sourceAlloc, copy);
      auto targetToken = tokenBefore(targetAlloc, copy);
      if (failed(sourceToken) || failed(targetToken)) return failure();

      auto copied = pustream::CopyOp::create(
        rewriter, loc, targetToken->getType(), *sourceToken, copy.getSource(), *targetToken, copy.getTarget()
      );
      pustream::AwaitOp::create(rewriter, loc, copied.getResult().getType(), copied.getResult());
      rewriter.eraseOp(copy);
      return success();
    }

    Value sram = sourceSram ? copy.getSource() : copy.getTarget();
    auto alloc = allocationOfView(sram);
    if (!alloc) return copy.emitError("cannot express this copy: the SRAM side does not trace back to an allocation");
    auto token = tokenBefore(alloc, copy);
    if (failed(token)) return failure();

    Value from       = sourceSram ? sram : copy.getSource();
    Value to         = sourceSram ? copy.getTarget() : sram;
    auto transferred = pustream::TransferOp::create(rewriter, loc, token->getType(), *token, from, to);
    pustream::AwaitOp::create(rewriter, loc, transferred.getResult().getType(), transferred.getResult());
    rewriter.eraseOp(copy);
    return success();
  }
};

class RuntimeLowering : public OpInterfaceRewritePattern<linalg::LinalgOp>
{
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  struct Kernel
  {
    StringRef callee;
    Type element;
    SmallVector<int64_t> templateArgs;
  };

  static std::optional<Kernel> kernelFor(linalg::LinalgOp op)
  {
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation())) {
      auto shape = matmulShape(matmul);
      if (!shape) return std::nullopt;
      return Kernel{"runtime_matmul", shape->element, {shape->M, shape->N, shape->K}};
    }
    if (auto transpose = dyn_cast<linalg::TransposeOp>(op.getOperation())) {
      auto shape = transposeShape(transpose);
      if (!shape) return std::nullopt;
      return Kernel{"runtime_transpose", shape->element, {shape->rows, shape->cols}};
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp op, PatternRewriter& rewriter) const override
  {
    auto kernel = kernelFor(op);
    if (!kernel) return failure();

    SmallVector<Value> ins, outs;
    if (!classify(op, ins, outs)) return failure();

    SmallVector<Value> insOperands, outPairs;
    for (Value value : ins)
      if (failed(appendPair(op, value, insOperands))) return failure();
    for (Value value : outs)
      if (failed(appendPair(op, value, outPairs))) return failure();

    SmallVector<Attribute> templateArgs{TypeAttr::get(kernel->element)};
    for (int64_t dim : kernel->templateArgs) templateArgs.push_back(rewriter.getIndexAttr(dim));

    emitDispatch(
      rewriter, op.getLoc(), FlatSymbolRefAttr::get(op.getContext(), kernel->callee),
      rewriter.getArrayAttr(templateArgs), insOperands, outPairs
    );
    rewriter.eraseOp(op);
    return success();
  }
};

class LinalgOpLowering : public OpInterfaceRewritePattern<linalg::LinalgOp>
{
public:
  LinalgOpLowering(MLIRContext* context, std::string outlinedAttr)
    : OpInterfaceRewritePattern(context), outlinedAttr_(std::move(outlinedAttr))
  {
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp op, PatternRewriter& rewriter) const override
  {
    if (RuntimeLowering::kernelFor(op)) return failure();
    if (!isComputable(op)) return failure();

    SmallVector<Value> ins, outs;
    if (!classify(op, ins, outs)) return failure();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc    = op.getLoc();

    SmallVector<Value> boundary(ins.begin(), ins.end());
    boundary.append(outs.begin(), outs.end());

    SmallVector<Value> insOperands, outPairs;
    for (Value value : ins) {
      if (!isa<MemRefType>(value.getType())) {
        insOperands.push_back(value);
        continue;
      }
      if (failed(appendPair(op, value, insOperands))) return failure();
    }
    for (Value value : outs)
      if (failed(appendPair(op, value, outPairs))) return failure();

    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());

    SmallVector<Type> types;
    for (Value value : boundary) types.push_back(value.getType());
    std::string name  = "outlined_" + std::to_string(index_++) + "_";
    std::string inner = op->getName().getStringRef().str();
    std::replace(inner.begin(), inner.end(), '.', '_');
    name += inner;

    auto function = func::FuncOp::create(builder, loc, name, builder.getFunctionType(types, TypeRange{}));
    SymbolTable::setSymbolVisibility(function, SymbolTable::Visibility::Private);
    function->setAttr(kOutlinedAttr, builder.getStringAttr(outlinedAttr_));

    Block* entry = function.addEntryBlock();
    IRMapping mapping;
    for (auto [value, argument] : llvm::zip_equal(boundary, entry->getArguments())) mapping.map(value, argument);
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(entry);
    bodyBuilder.clone(*op.getOperation(), mapping);
    func::ReturnOp::create(bodyBuilder, loc);

    emitDispatch(
      rewriter, loc, FlatSymbolRefAttr::get(getContext(), function.getSymName()), ArrayAttr{}, insOperands, outPairs
    );
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::string outlinedAttr_;
  mutable unsigned index_ = 0;
};

struct MemrefToPuStreamPass : public ::impl::MemrefToPuStreamBase<MemrefToPuStreamPass>
{
  using MemrefToPuStreamBase::MemrefToPuStreamBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<arith::ArithDialect, func::FuncDialect, pustream::PuStreamDialect>();
  }

  void runOnOperation() override
  {
    {
      RewritePatternSet allocations(&getContext());
      allocations.add<AllocOpLowering>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(allocations)))) {
        signalPassFailure();
        return;
      }
    }

    ConversionTarget target(getContext());
    target.addIllegalOp<memref::AllocOp, memref::CopyOp>();
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      if (auto linalg = dyn_cast<linalg::LinalgOp>(op)) return !isComputable(linalg);
      return true;
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<CopyOpLowering>(&getContext());
    patterns.add<RuntimeLowering>(&getContext());
    patterns.add<LinalgOpLowering>(&getContext(), outlinedAttr);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createMemrefToPuStreamPass()
{
  return std::make_unique<MemrefToPuStreamPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createMemrefToPuStreamPass(std::string outlinedAttr)
{
  MemrefToPuStreamOptions options;
  options.outlinedAttr = std::move(outlinedAttr);
  return std::make_unique<MemrefToPuStreamPass>(options);
}
