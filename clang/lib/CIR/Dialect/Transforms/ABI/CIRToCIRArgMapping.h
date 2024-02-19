#pragma once

#include "ABI/CIRFunctionInfo.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/ErrorHandling.h"
namespace mlir {
namespace cir {

/// Encapsulates information about the way function arguments from
/// LoweringFunctionInfo should be passed to actual CIR function.
class CIRToCIRArgMapping {

public:
  CIRToCIRArgMapping(const MLIRContext *ctx, const LoweringFunctionInfo &FI,
                     bool onlyRequiredArgs = false){};

  /// Returns index of first IR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getIRArgs(unsigned ArgNo) const {
    llvm_unreachable("NYI");
  }
};

} // namespace cir
} // namespace mlir
