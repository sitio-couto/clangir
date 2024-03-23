#include "ABIInfoImpl.h"
#include "CIRCXXABI.h"
#include "LoweringFunctionInfo.h"
#include "MissingFeature.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LoweringFunctionInfo &FI,
                        const ABIInfo &Info) {
  Type Ty = FI.getReturnType();

  if (const auto RT = Ty.dyn_cast<StructType>()) {
    assert(MissingFeature::isCXXRecord());
  }

  return CXXABI.classifyReturnType(FI);
}

Type useFirstFieldIfTransparentUnion(Type Ty) {
  if (auto RT = Ty.dyn_cast<StructType>()) {
    if (RT.isUnion())
      llvm_unreachable("NYI");
  }
  return Ty;
}

CIRCXXABI::RecordArgABI getRecordArgABI(const StructType RT, CIRCXXABI &CXXABI) {
  if (!MissingFeature::isCXXRecord()) {
    llvm_unreachable("NYI");
  }
  return CXXABI.getRecordArgABI(RT);
}

} // namespace cir
} // namespace mlir
