
#include "CIRCXXABI.h"
#include "LoweringModule.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

namespace {

class ItaniumCXXABI : public CIRCXXABI {

public:
  ItaniumCXXABI(LoweringModule &CGM) : CIRCXXABI(CGM) {}

  bool classifyReturnType(LoweringFunctionInfo &FI) const override;

  // FIXME(cir): This expects a CXXRecordDecl! Not any record type.
  RecordArgABI getRecordArgABI(const StructType RD) const override {
    // If C++ prohibits us from making a copy, pass by address.
    assert(MissingFeature::canPassInRegisters());
    return RAA_Default;
  }
};

} // namespace

bool ItaniumCXXABI::classifyReturnType(LoweringFunctionInfo &FI) const {
  const StructType RD = FI.getReturnType().dyn_cast<StructType>();
  if (!RD)
    return false;

  // If C++ prohibits us from making a copy, return by address.
  if (!MissingFeature::canPassInRegisters())
    llvm_unreachable("NYI");

  return false;
}

CIRCXXABI *CreateItaniumCXXABI(LoweringModule &CGM) {
  switch (CGM.getCXXABIKind()) {
  case clang::TargetCXXABI::GenericItanium:
    if (CGM.getTargetInfo().getTriple().getArch() == llvm::Triple::le32) {
      llvm_unreachable("NYI");
    }
    return new ItaniumCXXABI(CGM);

  case clang::TargetCXXABI::Microsoft:
    llvm_unreachable("Microsoft ABI is not Itanium-based");
  default:
    llvm_unreachable("NYI");
  }

  llvm_unreachable("bad ABI kind");
}

} // namespace cir
} // namespace mlir
