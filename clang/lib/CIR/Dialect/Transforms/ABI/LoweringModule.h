#pragma once

#include "CIRContext.h"
#include "LoweringTypes.h"
#include "MissingFeature.h"
#include "TargetLoweringInfo.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

// class CIRContext : llvm::RefCountedBase<CIRContext>{
// public:
//   CIRContext() = default;
//   ~CIRContext() = default;
// };

/// Replaces CodeGenModule from Clang in ABI lowering.
class LoweringModule {
private:
  CIRContext &context;
  ModuleOp module;
  const clang::TargetInfo &Target;
  mutable std::unique_ptr<TargetLoweringInfo> TheTargetCodeGenInfo;
  std::unique_ptr<CIRCXXABI> ABI;

  LoweringTypes types;

  PatternRewriter &rewriter;

public:
  LoweringModule(CIRContext &C, ModuleOp &module, StringAttr DL,
                 const clang::TargetInfo &target, PatternRewriter &rewriter);
  ~LoweringModule() = default;

  LoweringTypes &getTypes() { return types; }
  CIRContext &getContext() { return context; }
  CIRCXXABI &getCXXABI() const { return *ABI; }
  const clang::TargetInfo &getTarget() const { return Target; }
  const llvm::Triple &getTriple() const { return Target.getTriple(); }
  MLIRContext *getMLIRContext() { return module.getContext(); }
  ModuleOp &getModule() { return module; }

  const CIRDataLayout &getDataLayout() const { return types.getDataLayout(); }

  const TargetLoweringInfo &getTargetLoweringInfo();

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  const clang::TargetInfo &getTargetInfo() const { return Target; }

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  clang::TargetCXXABI::Kind getCXXABIKind() const {
    auto kind = getTarget().getCXXABI().getKind();
    assert(MissingFeature::langOpts());
    return kind;
  }

  void setCIRFunctionAttributes(FuncOp GD, const LoweringFunctionInfo &Info,
                                FuncOp F, bool IsThunk);

  void setFunctionAttributes(FuncOp GD, FuncOp F, bool IsIncompleteFunction,
                             bool IsThunk);

  FuncOp getOrCreateCIRFunction(
      StringRef MangledName, FuncType Ty, FuncOp D, bool ForVTable,
      bool DontDefer = false, bool IsThunk = false,
      ArrayAttr ExtraAttrs = {}, // TODO(cir): __attribute__(()) stuff.
      bool IsForDefinition = false);

  FuncOp getAddrOfFunction(FuncOp GD, FuncType Ty, bool ForVTable,
                           bool DontDefer, bool IsForDefinition);

  void rewriteGlobalFunctionDefinition(FuncOp op, LoweringModule &state,
                                       PatternRewriter &rewriter);
};

} // namespace cir
} // namespace mlir
