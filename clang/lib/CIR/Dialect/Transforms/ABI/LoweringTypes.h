#pragma once

// Used to replace CodeGenTypes from Clang in ABI lowering.
#include "ABI/CIRToCIRArgMapping.h"
#include "ABI/FnInfoOpts.h"
#include "ABI/LoweringFunctionInfo.h"
#include "ABI/MissingFeature.h"
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
class LoweringTypes;
class LoweringModule;
static const LoweringFunctionInfo &
arrangeLLVMFunctionInfo(LoweringTypes &CGT, bool instanceMethod,
                        SmallVectorImpl<mlir::Type> &prefix, FuncType FTP);

class LoweringTypes {
private:
  LoweringModule &LM;

  // FIXME(cir): We should be able to query this from LM.
  MLIRContext *ctx;
  MLIRContext *getContext() { return ctx; }

public:
  LoweringTypes(LoweringModule &LM, MLIRContext *ctx) : LM(LM), ctx(ctx){};
  ~LoweringTypes() = default;

  unsigned clangCallConvToLLVMCallConv(clang::CallingConv CC) {
    switch (CC) {
    case clang::CC_C:
      return llvm::CallingConv::C;
    default:
      llvm_unreachable("calling convention NYI");
    }
  }

  /// Arrange the argument and result information for a value of the
  /// given freestanding function type.
  const LoweringFunctionInfo &arrangeFreeFunctionType(FuncType FTy) {
    SmallVector<mlir::Type, 16> argTypes;
    return cir::arrangeLLVMFunctionInfo(*this, /*instanceMethod=*/false,
                                        argTypes, FTy);
  }

  /// Arrange the argument and result information for the declaration or
  /// definition of the given function.
  const LoweringFunctionInfo &arrangeFunctionDeclaration(FuncOp FD) {
    if (!MissingFeature::isMethod())
      llvm_unreachable("NYI");

    assert(MissingFeature::qualifiedTypes());
    FuncType FTy = FD.getFunctionType();

    assert(MissingFeature::CUDA());

    // When declaring a function without a prototype, always use a
    // non-variadic type.
    if (FD.getNoProto()) {
      llvm_unreachable("NYI");
    }

    return arrangeFreeFunctionType(FTy);
  }

  const LoweringFunctionInfo &arrangeGlobalDeclaration(FuncOp GD) {
    if (!MissingFeature::isCtorOrDtor())
      llvm_unreachable("NYI");

    return arrangeFunctionDeclaration(GD);
  }

  /// Arrange the argument and result information for an abstract value
  /// of a given function type.  This is the method which all of the
  /// above functions ultimately defer to.
  const LoweringFunctionInfo &arrangeLLVMFunctionInfo(Type resultType,
                                                      FnInfoOpts opts,
                                                      ArrayRef<Type> argTypes,
                                                      RequiredArgs required) {
    assert(MissingFeature::qualifiedTypes());

    assert(MissingFeature::fnInfoProfile());
    LoweringFunctionInfo *FI = nullptr;

    assert(MissingFeature::extParamInfo());
    unsigned CC = clangCallConvToLLVMCallConv(clang::CallingConv::CC_C);

    // Construct the function info.  We co-allocate the ArgInfos.
    FI = LoweringFunctionInfo::create(
        CC, /*isInstanceMethod=*/false, /*isChainCall=*/false,
        /*isDelegateCall=*/false, resultType, argTypes, required);

    assert(MissingFeature::recursiveFunctionProcessing());

    return *FI;
  }

  /// Return the ABI-specific function type for a CIR function type.
  FuncType getFunctionType(const LoweringFunctionInfo &FI) {

    assert(MissingFeature::recursiveFunctionProcessing());

    mlir::Type resultType = {};
    const ABIArgInfo &retAI = FI.getReturnInfo();
    switch (retAI.getKind()) {
    case ABIArgInfo::Direct:
      resultType = retAI.getCoerceToType();
      break;
    default:
      llvm_unreachable("Missing ABIArgInfo::Kind");
    }

    CIRToCIRArgMapping IRFunctionArgs(getContext(), FI, true);
    SmallVector<Type, 8> ArgTypes;

    // Add type for sret argument.
    assert(MissingFeature::sretArgument());

    // Add type for inalloca argument.
    assert(MissingFeature::inallocaArgument());

    // Add in all of the required arguments.
    unsigned ArgNo = 0;
    LoweringFunctionInfo::const_arg_iterator it = FI.arg_begin(),
                                             ie = it + FI.getNumRequiredArgs();
    for (; it != ie; ++it, ++ArgNo) {
      const ABIArgInfo &ArgInfo = it->info;

      assert(MissingFeature::argumentPadding());

      unsigned FirstIRArg, NumIRArgs;
      std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

      switch (ArgInfo.getKind()) {
      case ABIArgInfo::Direct: {
        // Fast-isel and the optimizer generally like scalar values better than
        // FCAs, so we flatten them if this is safe to do for this argument.
        Type argType = ArgInfo.getCoerceToType();
        StructType st = dyn_cast<StructType>(argType);
        if (st && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
          assert(NumIRArgs == st.getNumElements());
          for (unsigned i = 0, e = st.getNumElements(); i != e; ++i)
            ArgTypes[FirstIRArg + i] = st.getMembers()[i];
        } else {
          assert(NumIRArgs == 1);
          ArgTypes[FirstIRArg] = argType;
        }
        break;
      }
      default:
        llvm_unreachable("Missing ABIArgInfo::Kind");
      }
    }

    assert(MissingFeature::recursiveFunctionProcessing());

    return FuncType::get(getContext(), ArgTypes, resultType, FI.isVariadic());
  }
};

/// Adds the formal parameters in FPT to the given prefix. If any parameter in
/// FPT has pass_object_size attrs, then we'll add parameters for those, too.
static void appendParameterTypes(SmallVectorImpl<Type> &prefix, FuncType FPT) {
  // Fast path: don't touch param info if we don't need to.
  if (/*!FPT->hasExtParameterInfos()=*/true) {
    prefix.append(FPT.getInputs().begin(), FPT.getInputs().end());
    return;
  }

  assert(MissingFeature::extParamInfo());
  llvm_unreachable("NYI");
}

/// Arrange the LLVM function layout for a value of the given function
/// type, on top of any implicit parameters already stored.
static const LoweringFunctionInfo &
arrangeLLVMFunctionInfo(LoweringTypes &CGT, bool instanceMethod,
                        SmallVectorImpl<mlir::Type> &prefix, FuncType FTP) {
  assert(MissingFeature::extParamInfo());
  RequiredArgs Required = RequiredArgs::forPrototypePlus(FTP, prefix.size());
  // FIXME: Kill copy.
  appendParameterTypes(prefix, FTP);
  assert(MissingFeature::qualifiedTypes());
  Type resultType = FTP.getReturnType();

  FnInfoOpts opts =
      instanceMethod ? FnInfoOpts::IsInstanceMethod : FnInfoOpts::None;
  return CGT.arrangeLLVMFunctionInfo(resultType, opts, prefix, Required);
}

} // namespace cir
} // namespace mlir
