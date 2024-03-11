#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "CIRContext.h"
#include "LoweringTypes.h"
#include "MissingFeature.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

// Pin the vtable to this file.
ABIInfo::~ABIInfo() = default;

CIRCXXABI &ABIInfo::getCXXABI() const { return LT.getCXXABI(); }

CIRContext &ABIInfo::getContext() const { return LT.getContext(); }

const clang::TargetInfo &ABIInfo::getTarget() const { return LT.getTarget(); }

const CIRDataLayout &ABIInfo::getDataLayout() const {
  return LT.getDataLayout();
}

bool ABIInfo::isPromotableIntegerTypeForABI(Type Ty) const {
  if (getContext().isPromotableIntegerType(Ty))
    return true;

  assert(MissingFeature::fixedWidthIntegers());

  return false;
}

} // namespace cir
} // namespace mlir
