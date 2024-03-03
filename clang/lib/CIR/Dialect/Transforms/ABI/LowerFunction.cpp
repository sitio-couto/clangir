#include "LowerFunction.h"
#include "LoweringModule.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

LowerFunction::LowerFunction(LoweringModule &lm, PatternRewriter &rewriter)
    : LM(lm), Target(lm.getTarget()), rewriter(rewriter) {}

void LowerFunction::emitFunctionProlog(const LoweringFunctionInfo &FI,
                                       FuncOp Fn) {
  llvm_unreachable("Not implemented");
}

void LowerFunction::generateCode(FuncOp GD, FuncOp Fn,
                                 const LoweringFunctionInfo &FnInfo) {
  auto Args = GD.getArguments();
  Type ResTy = GD.getFunctionType().getReturnType();

  // NOTE(cir): Skipped some inline stuff from codegen here. Unlinkely that we
  // will need it for ABI lowering.

  // NOTE(cir): We may have to emit/edit function debug info here.

  startFunction(GD, ResTy, Fn, Args, FnInfo);
}

// Parity with CodeGenFunction::StartFunction. Note that the Fn variable is not
// a FuncOp, but a FuncType. In the original function, Fn is the result LLVM IR
// function, but here we are going to .
void LowerFunction::startFunction(FuncOp GD, Type RetTy, FuncOp Fn,
                                  llvm::MutableArrayRef<BlockArgument> &Args,
                                  const LoweringFunctionInfo &FnInfo) {
  // NOTE(cir): In the original Clang codegen, a lot of stuff is done here.
  // However, in CIR, we split this function between codegen and ABI lowering.
  // This means that the following sections are not necessary here as they will
  // be handled in CIR's codegen:
  // - Handling of sanitizers.
  // - Profiling.
  // - Addition/removal of function attributes.

  auto *entry = Fn.addEntryBlock();
  rewriter.setInsertionPointToEnd(entry);

  emitFunctionProlog(FnInfo, Fn);

  return;
}

} // namespace cir
} // namespace mlir
