#pragma once

#include "LoweringTypes.h"
#include "TargetLoweringInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

// Forward declarations.
static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LoweringModule &LM);

/// FIXME(cir): This should be moved to a more appropriate place.
/// The AVX ABI level for X86 targets.
enum class X86AVXABILevel {
  None,
  AVX,
  AVX512,
};

/// Replaces CodeGenModule from Clang in ABI lowering.
class LoweringModule {
private:
  ModuleOp module;
  const clang::TargetInfo &Target;
  mutable std::unique_ptr<TargetLoweringInfo> TheTargetCodeGenInfo;

  LoweringTypes types;

public:
  LoweringModule(ModuleOp &module, const clang::TargetInfo &target);
  ~LoweringModule() = default;

  LoweringTypes &getTypes() { return types; }
  MLIRContext *getContext() { return module.getContext(); }
  const clang::TargetInfo &getTarget() const { return Target; }
  const llvm::Triple &getTriple() const { return Target.getTriple(); }

  const TargetLoweringInfo &getTargetLoweringInfo();
};

} // namespace cir
} // namespace mlir
