#pragma once

#include "LoweringTypes.h"
#include "MissingFeature.h"
#include "TargetLoweringInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
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
  ModuleOp module;
  const clang::TargetInfo &Target;
  mutable std::unique_ptr<TargetLoweringInfo> TheTargetCodeGenInfo;
  std::unique_ptr<CIRCXXABI> ABI;

  LoweringTypes types;

public:
  LoweringModule(ModuleOp &module, const clang::TargetInfo &target);
  ~LoweringModule() = default;

  LoweringTypes &getTypes() { return types; }
  MLIRContext *getContext() { return module.getContext(); }
  CIRCXXABI &getCXXABI() const { return *ABI; }
  const clang::TargetInfo &getTarget() const { return Target; }
  const llvm::Triple &getTriple() const { return Target.getTriple(); }

  const TargetLoweringInfo &getTargetLoweringInfo();

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  const clang::TargetInfo &getTargetInfo() const { return Target; }

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  clang::TargetCXXABI::Kind getCXXABIKind() const {
    auto kind = getTarget().getCXXABI().getKind();
    assert(MissingFeature::langOpts());
    return kind;
  }
};

} // namespace cir
} // namespace mlir
