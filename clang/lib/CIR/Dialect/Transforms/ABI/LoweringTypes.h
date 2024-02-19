#pragma once

// Used to replace CodeGenTypes from Clang in ABI lowering.
#include "ABI/CIRFunctionInfo.h"
#include "ABI/LoweringModule.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

class LoweringModule;

class LoweringTypes {
private:
  LoweringModule &LM;

public:
  LoweringTypes(LoweringModule &LM) : LM(LM){};
  ~LoweringTypes() = default;

  const CIRFunctionInfo &arrangeGlobalDeclaration(FuncOp GD) {
    return {};
  }

  FuncType getFunctionType(const CIRFunctionInfo &FI) {
    return {};
  }
};

} // namespace cir
} // namespace mlir
