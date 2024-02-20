#include "ABIInfoImpl.h"
#include "CIRCXXABI.h"
#include "LoweringFunctionInfo.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LoweringFunctionInfo &FI,
                        const ABIInfo &Info) {
  llvm_unreachable("NYI");
}

} // namespace cir
} // namespace mlir
