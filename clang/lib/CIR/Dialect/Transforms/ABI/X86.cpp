#include "ABIInfo.h"
#include "LoweringFunctionInfo.h"
#include "LoweringModule.h"
#include "LoweringTypes.h"
#include "MissingFeature.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace mlir {
namespace cir {

class X86_64ABIInfo : public ABIInfo {
public:
  X86_64ABIInfo(LoweringTypes &CGT, X86AVXABILevel AVXLevel) : ABIInfo(CGT) {}

  void computeInfo(LoweringFunctionInfo &FI) const override;
};

class X86_64TargetLoweringInfo : public TargetLoweringInfo {
public:
  X86_64TargetLoweringInfo(LoweringTypes &CGT, X86AVXABILevel AVXLevel)
      : TargetLoweringInfo(std::make_unique<X86_64ABIInfo>(CGT, AVXLevel)) {
    assert(MissingFeature::Swift());
  }
};

void X86_64ABIInfo::computeInfo(LoweringFunctionInfo &FI) const {
  llvm_unreachable("X86_64ABIInfo::computeInfo is NYI");
}

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LoweringModule &CGM, X86AVXABILevel AVXLevel) {
  return std::make_unique<X86_64TargetLoweringInfo>(CGM.getTypes(), AVXLevel);
}

} // namespace cir
} // namespace mlir
