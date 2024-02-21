#include "ABIInfoImpl.h"
#include "CIRCXXABI.h"
#include "LoweringFunctionInfo.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LoweringFunctionInfo &FI,
                        const ABIInfo &Info) {
  Type Ty = FI.getReturnType();

  if (const auto RT = Ty.dyn_cast<StructType>()) {
    llvm_unreachable("NYI");
  }

  return CXXABI.classifyReturnType(FI);
}

} // namespace cir
} // namespace mlir
