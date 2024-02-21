#pragma once

#include "LoweringFunctionInfo.h"
namespace mlir {
namespace cir {

class LoweringModule;

// mlir::cir::LoweringModule;

class CIRCXXABI {
  friend class LoweringModule;

protected:
  LoweringModule &LM;

  CIRCXXABI(LoweringModule &LM) : LM(LM) {}

public:
  virtual ~CIRCXXABI();

  /// If the C++ ABI requires the given type be returned in a particular way,
  /// this method sets RetAI and returns true.
  virtual bool classifyReturnType(LoweringFunctionInfo &FI) const = 0;
};

/// Creates an Itanium-family ABI.
CIRCXXABI *CreateItaniumCXXABI(LoweringModule &CGM);

} // namespace cir
} // namespace mlir
