#include "LoweringTypes.h"
#include "LoweringCall.h"
#include "LoweringFunctionInfo.h"
#include "LoweringModule.h"
#include "MissingFeature.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Specifiers.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

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
///
/// \param CGT - Abstraction for lowering CIR types.
/// \param instanceMethod - Whether the function is an instance method.
/// \param prefix - List of implicit parameters to be prepended (e.g. 'this').
/// \param FTP - ABI-agnostic function type.
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

/// Arrange a call as unto a free function, except possibly with an
/// additional number of formal parameters considered required.
const LoweringFunctionInfo &
arrangeFreeFunctionLikeCall(LoweringTypes &CGT, LoweringModule &CGM,
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
  return CGT.arrangeLLVMFunctionInfo(fnType.getReturnType(), opts,
                                     fnType.getInputs(), required);
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

/// Figure out the rules for calling a function with the given formal
/// type using the given arguments.  The arguments are necessary
/// because the function might be unprototyped, in which case it's
/// target-dependent in crazy ways.
const LoweringFunctionInfo &
LoweringTypes::arrangeFreeFunctionCall(const OperandRange args,
                                       const FuncType fnType, bool chainCall) {
  return arrangeFreeFunctionLikeCall(*this, LM, args, fnType, chainCall ? 1 : 0,
                                     chainCall);
}

const LoweringFunctionInfo &LoweringTypes::arrangeGlobalDeclaration(FuncOp GD) {
  if (isa<clang::CXXConstructorDecl>(GD.getDecl()) ||
      isa<clang::CXXDestructorDecl>(GD.getDecl()))
    llvm_unreachable("NYI");

  return arrangeFunctionDeclaration(GD);
}

/// Convert a CIR type to its ABI-specific default form.
///
/// NOTE(cir): It the original codegen this method is used to get the default
/// LLVM IR representation for a given AST type. When a the ABI-specific
/// function info sets a nullptr for a return or argument type, the default type
/// given by this method is used. I think we can move this logic into the
/// LoweringFunctionInfo class, but for the sake of parity, I'm keeping it here
/// for now.
Type LoweringTypes::convertType(Type T) {
  // Certain CIR types are already ABI-specific, so we just return them.
  if (T.isa<IntType, FloatType>()) {
    return T;
  }

  // Default bool type to `!cir.int<u, 1>`.
  if (T.isa<BoolType>()) {
    return IntType::get(getMLIRContext(), /*width=*/1, /*isSigned=*/false);
  }

  llvm::outs() << "Missing default ABI-specific type for " << T << "\n";
  llvm_unreachable("NYI");
}

/// Arrange the argument and result information for an abstract value
/// of a given function type.  This is the method which all of the
/// above functions ultimately defer to.
///
/// \param resultType - ABI-agnostic CIR result type.
/// \param opts - Options to control the arrangement.
/// \param argTypes - ABI-agnostic CIR argument types.
/// \param required - Information about required/optional arguments.
const LoweringFunctionInfo &
LoweringTypes::arrangeLLVMFunctionInfo(Type resultType, FnInfoOpts opts,
                                       ArrayRef<Type> argTypes,
                                       RequiredArgs required) {
  assert(MissingFeature::qualifiedTypes());

  assert(MissingFeature::fnInfoProfile());
  LoweringFunctionInfo *FI = nullptr;

  // FIXME(cir): Users may enforce a specific CC for a function using, for
  // example: void __attribute__((vectorcall)) func(int a) {}. CIR functions
  // should carry this information so that we may query it here instead of
  // always passing CC_C.
  assert(MissingFeature::extParamInfo());
  unsigned CC = clangCallConvToLLVMCallConv(clang::CallingConv::CC_C);

  // Construct the function info. We co-allocate the ArgInfos.
  // NOTE(cir): The initial function info might hold incorrect data.
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
    // NOTE(cir): Properly compute function info patching any incorrect data.
    getABIInfo().computeInfo(*FI); // FIXME(cir): Args should be set to null.
  }

  // Loop over all of the computed argument and return value info. If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ABIArgInfo &retInfo = FI->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    retInfo.setCoerceToType(convertType(FI->getReturnType()));

  for (auto &I : FI->arguments())
    if (I.info.canHaveCoerceToType() && I.info.getCoerceToType() == nullptr)
      I.info.setCoerceToType(convertType(I.type));

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
