#pragma once

#include "ABI/LoweringTypes.h"
#include "ABI/TargetLoweringInfo.h"
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
  LoweringTypes types;
  const clang::TargetInfo &Target;

  mutable std::unique_ptr<TargetLoweringInfo> TheTargetCodeGenInfo;

public:
  LoweringModule(ModuleOp &module, const clang::TargetInfo &target)
      : module(module), types(*this, module.getContext()), Target(target){};
  ~LoweringModule() = default;

  LoweringTypes &getTypes() { return types; }
  MLIRContext *getContext() { return module.getContext(); }
  const clang::TargetInfo &getTarget() const { return Target; }
  const llvm::Triple &getTriple() const { return Target.getTriple(); }

  const TargetLoweringInfo &getTargetLoweringInfo() {
    if (!TheTargetCodeGenInfo)
      TheTargetCodeGenInfo = createTargetLoweringInfo(*this);
    return *TheTargetCodeGenInfo;
  }
};

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LoweringModule &LM) {
  const clang::TargetInfo &Target = LM.getTarget();
  const llvm::Triple &Triple = Target.getTriple();

  switch (Triple.getArch()) {
  case llvm::Triple::x86_64: {
    StringRef ABI = Target.getABI();
    X86AVXABILevel AVXLevel = (ABI == "avx512" ? X86AVXABILevel::AVX512
                               : ABI == "avx"  ? X86AVXABILevel::AVX
                                               : X86AVXABILevel::None);

    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
      llvm_unreachable("Windows ABI NYI");
    default:
      llvm_unreachable("ABI NYI");
    }
  }
  default:
    llvm_unreachable("ABI NYI");
  }
}

} // namespace cir
} // namespace mlir
