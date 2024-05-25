#include "LowerCall.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"

using namespace mlir;
using namespace cir;

namespace {

/// Arrange a call as unto a free function, except possibly with an
/// additional number of formal parameters considered required.
const LowerFunctionInfo &
arrangeFreeFunctionLikeCall(LowerTypes &LT, LowerModule &LM,
                            const OperandRange &args, const FuncType fnType,
                            unsigned numExtraRequiredArgs, bool chainCall) {
  assert(args.size() >= numExtraRequiredArgs);

  assert(MissingFeature::extParamInfo());

  // In most cases, there are no optional arguments.
  RequiredArgs required = RequiredArgs::All;

  // If we have a variadic prototype, the required arguments are the
  // extra prefix plus the arguments in the prototype.
  // FIXME(cir): We need a way to check for no-proto function calls. In CIR,
  // it's the function op that carries the no-proto flag, not the type. We could
  // keep a symbol table here to query the func op, but it might create
  // concurrency issues. This function is also missing the CallOp to check for
  // no-proto calls.
  if (/*IsPrototypedFunction=*/true) {
    if (fnType.isVarArg())
      llvm_unreachable("NYI");

    if (!MissingFeature::extParamInfo())
      llvm_unreachable("NYI");
  }

  // NOTE(cir): There's some CC stuff related to no-proto functions here, but
  // I'm skipping it since it requires CodeGen info. Maybe we can embbed this
  // information in the FuncOp during CIRGen.

  // NOTE(cir): There would be a loop here to get the canonical form of each AST
  // arg, but this is not necessary in CIR right now.
  assert(MissingFeature::chainCall() && !chainCall && "NYI");
  FnInfoOpts opts = chainCall ? FnInfoOpts::IsChainCall : FnInfoOpts::None;
  return LT.arrangeLLVMFunctionInfo(fnType.getReturnType(), opts,
                                    fnType.getInputs(), required);
}

} // namespace

/// Figure out the rules for calling a function with the given formal
/// type using the given arguments.  The arguments are necessary
/// because the function might be unprototyped, in which case it's
/// target-dependent in crazy ways.
const LowerFunctionInfo &
LowerTypes::arrangeFreeFunctionCall(const OperandRange args,
                                    const FuncType fnType, bool chainCall) {
  return arrangeFreeFunctionLikeCall(*this, LM, args, fnType, chainCall ? 1 : 0,
                                     chainCall);
}

/// Arrange the argument and result information for an abstract value
/// of a given function type.  This is the method which all of the
/// above functions ultimately defer to.
///
/// \param resultType - ABI-agnostic CIR result type.
/// \param opts - Options to control the arrangement.
/// \param argTypes - ABI-agnostic CIR argument types.
/// \param required - Information about required/optional arguments.
const LowerFunctionInfo &
LowerTypes::arrangeLLVMFunctionInfo(Type resultType, FnInfoOpts opts,
                                    ArrayRef<Type> argTypes,
                                    RequiredArgs required) {
  assert(MissingFeature::qualifiedTypes());

  // assert(MissingFeature::fnInfoProfile());
  LowerFunctionInfo *FI = nullptr;

  // FIXME(cir): Users may enforce a specific CC for a function using, for
  // example: void __attribute__((vectorcall)) func(int a) {}. CIR functions
  // should carry this information so that we may query it here instead of
  // always passing CC_C.
  assert(MissingFeature::extParamInfo());
  unsigned CC = clangCallConvToLLVMCallConv(clang::CallingConv::CC_C);

  // Construct the function info. We co-allocate the ArgInfos.
  // NOTE(cir): The initial function info might hold incorrect data.
  FI = LowerFunctionInfo::create(
      CC, /*isInstanceMethod=*/false, /*isChainCall=*/false,
      /*isDelegateCall=*/false, resultType, argTypes, required);

  // assert(MissingFeature::recursiveFunctionProcessing());

  // Compute ABI information.
  if (CC == llvm::CallingConv::SPIR_KERNEL) {
    llvm_unreachable("NYI");
  } else if (!MissingFeature::extParamInfo()) {
    llvm_unreachable("NYI");
  } else {
    // NOTE(cir): Properly compute function info patching any incorrect data.
    getABIInfo().computeInfo(*FI); // FIXME(cir): Args should be set to null.
  }

  // Loop over all of the computed argument and return value info. If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ABIArgInfo &retInfo = FI->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    llvm_unreachable("NYI");

  for (auto &_ : FI->arguments())
    llvm_unreachable("NYI");

  // assert(MissingFeature::recursiveFunctionProcessing());

  return *FI;
}
