#pragma once

#include "ABIInfo.h"
#include "llvm/Support/ErrorHandling.h"
namespace mlir {
namespace cir {

class TargetLoweringInfo {
private:
public:
  TargetLoweringInfo() = default;
  ~TargetLoweringInfo() = default;

  const ABIInfo &getABIInfo() const {
    llvm_unreachable("Not implemented");  
  }
};
  
} // namespace cir
} // namespace mlir
