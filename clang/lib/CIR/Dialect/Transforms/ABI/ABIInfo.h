#pragma once

#include "LoweringFunctionInfo.h"

namespace mlir {
namespace cir {

/// ABIInfo - Target specific hooks for defining how a type should be
/// passed or returned from functions.
class ABIInfo {
  virtual void computeInfo(LoweringFunctionInfo &FI) const = 0;
};

} // namespace cir
} // namespace mlir
