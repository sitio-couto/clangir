#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

// Pin the vtable to this file.
ABIInfo::~ABIInfo() = default;

CIRCXXABI &ABIInfo::getCXXABI() const { 
  llvm_unreachable("NYI");
}

} // namespace cir
} // namespace mlir
