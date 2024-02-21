#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "CIRContext.h"
#include "LoweringTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

// Pin the vtable to this file.
ABIInfo::~ABIInfo() = default;

CIRCXXABI &ABIInfo::getCXXABI() const { return LT.getCXXABI(); }

CIRContext &ABIInfo::getContext() const { return LT.getContext(); }

} // namespace cir
} // namespace mlir
