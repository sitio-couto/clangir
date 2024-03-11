#pragma once

#include "CIRCXXABI.h"
#include "CIRContext.h"
#include "DataLayout.h"
#include "mlir/IR/MLIRContext.h"
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

  CIRCXXABI &getCXXABI() const;
  CIRContext &getContext() const;
  const clang::TargetInfo &getTarget() const;
  const CIRDataLayout &getDataLayout() const;

  virtual void computeInfo(LoweringFunctionInfo &FI) const = 0;

  // Implement the Type::IsPromotableIntegerType for ABI specific needs. The
  // only difference is that this considers bit-precise integer types as well.
  bool isPromotableIntegerTypeForABI(Type Ty) const;
};

} // namespace cir
} // namespace mlir
