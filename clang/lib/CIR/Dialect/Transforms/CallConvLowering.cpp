//===- CallConvLowering.cpp - Rewrites functions conforming to call convs -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME(cir): This header file is not exposed to the public API, but can be
// reused by CIR ABI lowering, since it holds target-specific information.
#include "../../../Basic/Targets.h"

#include "ABI/CIRContext.h"
#include "ABI/LoweringFunctionInfo.h"
#include "ABI/LoweringModule.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

struct DummyRewrite : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<mlir::ModuleOp>();

    llvm::Triple triple(
        module->getAttr("cir.triple").cast<StringAttr>().getValue());
    clang::TargetOptions targetOptions;
    targetOptions.Triple = triple.str();

    auto targetInfo = clang::targets::AllocateTarget(triple, targetOptions);

    auto context = CIRContext();
    context.initBuiltinTypes(*targetInfo);

    LoweringModule state(context, module, *targetInfo);

    const LoweringFunctionInfo &FI =
        state.getTypes().arrangeGlobalDeclaration(op);
    FuncType Ty = state.getTypes().getFunctionType(FI);
    Ty.dump();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CallConvLoweringPass
    : ::impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  void runOnOperation() override;
  StringRef getArgument() const override { return "call-conv-lowering"; };
};

void populateCallConvLoweringPassPatterns(RewritePatternSet &patterns) {
  patterns.add<DummyRewrite>(patterns.getContext());
}

void CallConvLoweringPass::runOnOperation() {

  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateCallConvLoweringPassPatterns(patterns);

  // Collect operations to apply pattern.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    if (isa<FuncOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns))))
    signalPassFailure();
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
