#include "LoweringTypes.h"
#include "LoweringModule.h"
#include "clang/Basic/Specifiers.h"

using namespace mlir;
using namespace cir;

namespace {

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

} // namespace

LoweringTypes::LoweringTypes(LoweringModule &LM, StringRef DLString)
    : LM(LM), context(LM.getContext()), Target(LM.getTarget()),
      CXXABI(LM.getCXXABI()),
      TheABIInfo(LM.getTargetLoweringInfo().getABIInfo()),
      mlirContext(LM.getMLIRContext()), DL(DLString, LM.getModule()) {}

unsigned LoweringTypes::clangCallConvToLLVMCallConv(clang::CallingConv CC) {
  switch (CC) {
  case clang::CC_C:
    return llvm::CallingConv::C;
  default:
    llvm_unreachable("calling convention NYI");
  }
}

/// Arrange the argument and result information for a value of the
/// given freestanding function type.
const LoweringFunctionInfo &
LoweringTypes::arrangeFreeFunctionType(FuncType FTy) {
  SmallVector<mlir::Type, 16> argTypes;
  return ::arrangeLLVMFunctionInfo(*this, /*instanceMethod=*/false, argTypes,
                                   FTy);
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const LoweringFunctionInfo &
LoweringTypes::arrangeFunctionDeclaration(FuncOp FD) {
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

const LoweringFunctionInfo &LoweringTypes::arrangeGlobalDeclaration(FuncOp GD) {
  if (!MissingFeature::isCtorOrDtor())
    llvm_unreachable("NYI");

  return arrangeFunctionDeclaration(GD);
}

/// Arrange the argument and result information for an abstract value
/// of a given function type.  This is the method which all of the
/// above functions ultimately defer to.
const LoweringFunctionInfo &
LoweringTypes::arrangeLLVMFunctionInfo(Type resultType, FnInfoOpts opts,
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

  // Compute ABI information.
  if (CC == llvm::CallingConv::SPIR_KERNEL) {
    llvm_unreachable("NYI");
  } else if (!MissingFeature::extParamInfo()) {
    llvm_unreachable("NYI");
  } else {
    getABIInfo().computeInfo(*FI);
  }

  // Loop over all of the computed argument and return value info. If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ABIArgInfo &retInfo = FI->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    retInfo.setCoerceToType(FI->getReturnType());

  for (auto &I : FI->arguments())
    if (I.info.canHaveCoerceToType() && I.info.getCoerceToType() == nullptr)
      I.info.setCoerceToType(I.type);

  assert(MissingFeature::recursiveFunctionProcessing());

  return *FI;
}

/// Return the ABI-specific function type for a CIR function type.
FuncType LoweringTypes::getFunctionType(const LoweringFunctionInfo &FI) {

  assert(MissingFeature::recursiveFunctionProcessing());

  mlir::Type resultType = {};
  const ABIArgInfo &retAI = FI.getReturnInfo();
  switch (retAI.getKind()) {
  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    resultType = retAI.getCoerceToType();
    break;
  case ABIArgInfo::Ignore:
    resultType = VoidType::get(getMLIRContext());
    break;
  default:
    llvm_unreachable("Missing ABIArgInfo::Kind");
  }

  CIRToCIRArgMapping IRFunctionArgs(getContext(), FI, true);
  SmallVector<Type, 8> ArgTypes(IRFunctionArgs.totalIRArgs());

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
    case ABIArgInfo::Extend:
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

  return FuncType::get(getMLIRContext(), ArgTypes, resultType, FI.isVariadic());
}
