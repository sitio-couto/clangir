#pragma once

#include "CIRCXXABI.h"
#include "mlir/IR/PatternMatch.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

class LowerFunction {
  LowerFunction(const LowerFunction &) = delete;
  void operator=(const LowerFunction &) = delete;

  friend class CIRCXXABI;

  LoweringModule &LM; // Per-module state.
  const clang::TargetInfo &Target;

  PatternRewriter &rewriter;

public:
  LowerFunction(LoweringModule &cgm, PatternRewriter &rewriter);
  ~LowerFunction() = default;

  void emitFunctionProlog(const LoweringFunctionInfo &FI, FuncOp Fn);

  // Parity with CodeGenFunction::StartFunction. Note that the Fn variable is
  // not a FuncOp, but a FuncType. In the original function, Fn is the result
  // LLVM IR function, but here we are going to .
  void startFunction(FuncOp GD, Type RetTy, FuncOp Fn,
                     llvm::MutableArrayRef<BlockArgument> &Args,
                     const LoweringFunctionInfo &FnInfo);

  // Parity with CodeGenFunction::GenerateCode. Keep in mind that several
  // sections in the original function are focused on codegen unrelated to the
  // ABI. Such sections are handled in CIR's codegen, not here.
  void generateCode(FuncOp GD, FuncOp Fn, const LoweringFunctionInfo &FnInfo);
};

} // namespace cir
} // namespace mlir
