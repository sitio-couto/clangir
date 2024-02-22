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

  /// Specify how one should pass an argument of a record type.
  enum RecordArgABI {
    /// Pass it using the normal C aggregate rules for the ABI, potentially
    /// introducing extra copies and passing some or all of it in registers.
    RAA_Default = 0,

    /// Pass it on the stack using its defined layout.  The argument must be
    /// evaluated directly into the correct stack position in the arguments
    /// area,
    /// and the call machinery must not move it or introduce extra copies.
    RAA_DirectInMemory,

    /// Pass it as a pointer to temporary memory.
    RAA_Indirect
  };

  /// Returns how an argument of the given record type should be passed.
  /// FIXME(cir): This expects a CXXRecordDecl! Not any record type.
  virtual RecordArgABI getRecordArgABI(const StructType RD) const = 0;
};

/// Creates an Itanium-family ABI.
CIRCXXABI *CreateItaniumCXXABI(LoweringModule &CGM);

} // namespace cir
} // namespace mlir
