#pragma once

// Used to replace CodeGenTypes from Clang in ABI lowering.
#include "ABIInfo.h"
#include "CIRContext.h"
#include "CIRToCIRArgMapping.h"
#include "DataLayout.h"
#include "LoweringCall.h"
#include "LoweringFunctionInfo.h"
#include "MissingFeature.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "clang/Basic/Specifiers.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

// Forward declarations.
class LoweringModule;

// Used to replace CodeGenTypes from Clang in ABI lowering.
class LoweringTypes {
private:
  LoweringModule &LM;
  CIRContext &context;
  const clang::TargetInfo &Target;
  CIRCXXABI &CXXABI;

  // This should not be moved earlier, since its initialization depends on some
  // of the previous reference members being already initialized
  const ABIInfo &TheABIInfo;

  // Used to build types and other MLIR operations.
  MLIRContext *mlirContext;

  CIRDataLayout DL;

  const ABIInfo &getABIInfo() const { return TheABIInfo; }

public:
  LoweringTypes(LoweringModule &LM, StringRef DLString);
  ~LoweringTypes() = default;

  const CIRDataLayout &getDataLayout() const { return DL; }
  LoweringModule &getCGM() const { return LM; }
  CIRContext &getContext() { return context; }
  const clang::TargetInfo &getTarget() const { return Target; }
  CIRCXXABI &getCXXABI() const { return CXXABI; }
  MLIRContext *getMLIRContext() { return mlirContext; }

  unsigned clangCallConvToLLVMCallConv(clang::CallingConv CC);

  /// Arrange the argument and result information for a value of the
  /// given freestanding function type.
  const LoweringFunctionInfo &arrangeFreeFunctionType(FuncType FTy);

  const LoweringFunctionInfo &arrangeGlobalDeclaration(FuncOp GD);

  const LoweringFunctionInfo &arrangeFunctionDeclaration(FuncOp FD);

  /// Arrange the argument and result information for an abstract value
  /// of a given function type.  This is the method which all of the
  /// above functions ultimately defer to.
  const LoweringFunctionInfo &arrangeLLVMFunctionInfo(Type resultType,
                                                      FnInfoOpts opts,
                                                      ArrayRef<Type> argTypes,
                                                      RequiredArgs required);

  /// Return the ABI-specific function type for a CIR function type.
  FuncType getFunctionType(const LoweringFunctionInfo &FI);
};

} // namespace cir
} // namespace mlir
