#pragma once

#include "ABIInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
namespace mlir {
namespace cir {

class TargetLoweringInfo {
private:
  std::unique_ptr<ABIInfo> Info;

public:
  TargetLoweringInfo(std::unique_ptr<ABIInfo> Info);
  virtual ~TargetLoweringInfo();

  const ABIInfo &getABIInfo() const { return *Info; }
};

} // namespace cir
} // namespace mlir
