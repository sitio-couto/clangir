#pragma once

#include "llvm/IR/CallingConv.h"

namespace mlir {
namespace cir {

// Forward declarations.
class LoweringFunctionInfo;
class LoweringTypes;

/// ABIInfo - Target specific hooks for defining how a type should be
/// passed or returned from functions.
class ABIInfo {
protected:
  LoweringTypes &LT;
  llvm::CallingConv::ID RuntimeCC;

public:
  ABIInfo(LoweringTypes &LT) : LT(LT), RuntimeCC(llvm::CallingConv::C) {}
  virtual ~ABIInfo();

  virtual void computeInfo(LoweringFunctionInfo &FI) const = 0;
};

} // namespace cir
} // namespace mlir
