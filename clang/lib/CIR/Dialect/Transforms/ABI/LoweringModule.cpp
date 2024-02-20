#include "LoweringModule.h"

namespace mlir {
namespace cir {

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

LoweringModule::LoweringModule(ModuleOp &module,
                               const clang::TargetInfo &target)
    : module(module), Target(target), types(*this) {}

const TargetLoweringInfo &LoweringModule::getTargetLoweringInfo() {
  if (!TheTargetCodeGenInfo)
    TheTargetCodeGenInfo = createTargetLoweringInfo(*this);
  return *TheTargetCodeGenInfo;
}

} // namespace cir
} // namespace mlir
