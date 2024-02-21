#include "LoweringModule.h"
#include "CIRContext.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

static CIRCXXABI *createCXXABI(LoweringModule &CGM) {
  switch (CGM.getCXXABIKind()) {
  case clang::TargetCXXABI::AppleARM64:
  case clang::TargetCXXABI::Fuchsia:
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::GenericARM:
  case clang::TargetCXXABI::iOS:
  case clang::TargetCXXABI::WatchOS:
  case clang::TargetCXXABI::GenericMIPS:
  case clang::TargetCXXABI::GenericItanium:
  case clang::TargetCXXABI::WebAssembly:
  case clang::TargetCXXABI::XL:
    return CreateItaniumCXXABI(CGM);
  case clang::TargetCXXABI::Microsoft:
    llvm_unreachable("Windows ABI NYI");
  }

  llvm_unreachable("invalid C++ ABI kind");
}

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LoweringModule &LM) {
  const clang::TargetInfo &Target = LM.getTarget();
  const llvm::Triple &Triple = Target.getTriple();

  switch (Triple.getArch()) {
  case llvm::Triple::x86_64: {
    // StringRef ABI = Target.getABI();
    // X86AVXABILevel AVXLevel = (ABI == "avx512" ? X86AVXABILevel::AVX512
    //                            : ABI == "avx"  ? X86AVXABILevel::AVX
    //                                            : X86AVXABILevel::None);

    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
      llvm_unreachable("Windows ABI NYI");
    default:
      return createX86_64TargetLoweringInfo(LM, X86AVXABILevel::None);
    }
  }
  default:
    llvm_unreachable("ABI NYI");
  }
}

LoweringModule::LoweringModule(CIRContext &C, ModuleOp &module,
                               const clang::TargetInfo &target)
    : context(C), module(module), Target(target), ABI(createCXXABI(*this)),
      types(*this) {}

const TargetLoweringInfo &LoweringModule::getTargetLoweringInfo() {
  if (!TheTargetCodeGenInfo)
    TheTargetCodeGenInfo = createTargetLoweringInfo(*this);
  return *TheTargetCodeGenInfo;
}

} // namespace cir
} // namespace mlir
