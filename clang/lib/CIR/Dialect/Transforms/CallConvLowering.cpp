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
#include "ABI/CIRToCIRArgMapping.h"
#include "ABI/LoweringFunctionInfo.h"
#include "ABI/LoweringModule.h"
#include "ABI/MissingFeature.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

struct CallConvFuncDefRewrite : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.getAst() && "missing AST information");

    // FIXME(cir): Organize the whole LoweringModule initialization.
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto dataLayout =
        module->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName())
            .cast<StringAttr>();

    llvm::Triple triple(
        module->getAttr("cir.triple").cast<StringAttr>().getValue());
    clang::TargetOptions targetOptions;
    targetOptions.Triple = triple.str();

    auto targetInfo = clang::targets::AllocateTarget(triple, targetOptions);

    // FIXME(cir): This just uses the default language options.
    assert(MissingFeature::langOptions());
    clang::LangOptions langOpts;

    auto context = CIRContext(module.getContext(), langOpts);
    context.initBuiltinTypes(*targetInfo);

    LoweringModule state(context, module, dataLayout, *targetInfo, rewriter);

    // TODO(cir): Would be nice to separate FuncOp and CallOp rewrites, but a
    // few complications come with it:
    //  - If a function is rewritten before its call, we loose ABI-agnostic
    //    information useful to lower the CallOp.
    //  - We might loose some cacheing benefits. AFAIK, there is no way to share
    //    a object across all patterns in a pass.

    // Rewrite function calls before definitions. We need to have the info from
    // the original function to properly lower its calls.
    auto calls = op.getSymbolUses(module);
    if (calls.has_value())
      for (auto call : calls.value())
        state.rewriteFunctionCall(cast<CallOp>(call.getUser()), op);

    // TODO(cir): Instead of re-emmiting loads and stores, just bitcast
    // arguments and return values to their ABI-specific counterparts.
    // TODO(cir): We also need to properly replace value uses and erase the old
    // operations.
    // TODO(cir): Fix arguments & return val attributes printing/parsing.
    state.rewriteGlobalFunctionDefinition(op, state, rewriter);

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
  patterns.add<CallConvFuncDefRewrite>(patterns.getContext());
}

void CallConvLoweringPass::runOnOperation() {
  assert(getOperation()->getAttr("cir.triple") && "missing target triple");

  // Collect rewrite patterns to be used in the pass.
  RewritePatternSet patterns(&getContext());
  populateCallConvLoweringPassPatterns(patterns);

  // Collect operations to be considered by the pass.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    if (isa<FuncOp, CallOp>(op))
      ops.push_back(op);
  });

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  // Apply patterns to selected ops.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
