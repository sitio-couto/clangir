#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "LoweringTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

// Pin the vtable to this file.
ABIInfo::~ABIInfo() = default;

CIRCXXABI &ABIInfo::getCXXABI() const { return LT.getCXXABI(); }

} // namespace cir
} // namespace mlir
