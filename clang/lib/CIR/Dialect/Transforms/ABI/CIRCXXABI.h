#pragma once

namespace mlir {
namespace cir {

class LoweringModule;

// mlir::cir::LoweringModule;

class CIRCXXABI {
  friend class LoweringModule;

protected:
  LoweringModule &LM;

public:
  CIRCXXABI(LoweringModule &LM) : LM(LM) {}
};

/// Creates an Itanium-family ABI.
CIRCXXABI *CreateItaniumCXXABI(LoweringModule &CGM);

} // namespace cir
} // namespace mlir
