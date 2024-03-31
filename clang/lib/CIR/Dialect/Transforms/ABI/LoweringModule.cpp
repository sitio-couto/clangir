#include "LoweringModule.h"
#include "CIRContext.h"
#include "CIRToCIRArgMapping.h"
#include "LowerFunction.h"
#include "LoweringFunctionInfo.h"
#include "MissingFeature.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>

namespace mlir {
namespace cir {

static CIRCXXABI *createCXXABI(LoweringModule &CGM) {
  switch (CGM.getCXXABIKind()) {
  case clang::TargetCXXABI::AppleARM64:
  case clang::TargetCXXABI::Fuchsia:
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::GenericARM:
  case clang::TargetCXXABI::iOS:
  case clang::TargetCXXABI::WatchOS:
  case clang::TargetCXXABI::GenericMIPS:
  case clang::TargetCXXABI::GenericItanium:
  case clang::TargetCXXABI::WebAssembly:
  case clang::TargetCXXABI::XL:
    return CreateItaniumCXXABI(CGM);
  case clang::TargetCXXABI::Microsoft:
    llvm_unreachable("Windows ABI NYI");
  }

  llvm_unreachable("invalid C++ ABI kind");
}

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LoweringModule &LM) {
  const clang::TargetInfo &Target = LM.getTarget();
  const llvm::Triple &Triple = Target.getTriple();

  switch (Triple.getArch()) {
  case llvm::Triple::x86_64: {
    // StringRef ABI = Target.getABI();
    // X86AVXABILevel AVXLevel = (ABI == "avx512" ? X86AVXABILevel::AVX512
    //                            : ABI == "avx"  ? X86AVXABILevel::AVX
    //                                            : X86AVXABILevel::None);

    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
      llvm_unreachable("Windows ABI NYI");
    default:
      return createX86_64TargetLoweringInfo(LM, X86AVXABILevel::None);
    }
  }
  default:
    llvm_unreachable("ABI NYI");
  }
}

LoweringModule::LoweringModule(CIRContext &C, ModuleOp &module, StringAttr DL,
                               const clang::TargetInfo &target,
                               PatternRewriter &rewriter)
    : context(C), module(module), Target(target), ABI(createCXXABI(*this)),
      types(*this, DL.getValue()), rewriter(rewriter) {}

const TargetLoweringInfo &LoweringModule::getTargetLoweringInfo() {
  if (!TheTargetCodeGenInfo)
    TheTargetCodeGenInfo = createTargetLoweringInfo(*this);
  return *TheTargetCodeGenInfo;
}

/// Construct the ABI-specific IR attribute list of a function or call.
///
/// NOTE(cir): This method copies CodeGenModule::ConstructAttributeList, but
/// only partially. The parts copied here are the ones focussed on calling
/// conventions that direclty affect a function's signature. For example, the
/// zero/signext parameter attributes that indicate if a type has to be extended
/// to the target's architecture lenght. With this in mind, before adding an
/// attribute, consider where it should be handled: In CIRGen, if
/// codegen-related; here, if call conv related; or in a different pass, if not
/// codegen and call conv related.
///
/// NOTE(cir): Another notable difference is that we are not using an
/// AttributeList abstraction like the original method. Instead, we are binding
/// the attributes directly to the function as we go through MLIR's
/// FunctionOpInterface.
void LoweringModule::constructAttributeList(
    StringRef Name, const LoweringFunctionInfo &FI,
    FuncOp CalleeInfo, // TODO(cir): Implement CalleeInfo class?
    FuncOp newFn, unsigned &CallingConv, bool AttrOnCallSite, bool IsThunk) {
  SmallVector<Attribute> FuncAttrs;
  // SmallVector<Attribute> RetAttrs;

  CallingConv = FI.getCallingConvention();

  // NOTE(cir): We will skip a lot of attributes added by the original method
  // since they are mostly related codegen and would be better handled in
  // CIRGen.

  // TODO(cir): Implement CSME NS Call attribute for ARM. This is an external
  // attribute (e.g. __attribute__), I think.

  // FIXME(vini): Improve this comment.
  // TODO(cir): AddAttributesFromFunctionProtoType
  // TODO(cir): AddAttributesFromAssumes

  // // Some ABIs may result in additional accesses to arguments that may
  // // otherwise not be present.
  // auto AddPotentialArgAccess = [&]() {
  //   llvm_unreachable("AddPotentialArgAccess is NYI");
  // };

  // TODO(cir): Regarding CodeGenModule::getDefaultFunctionAttributes:
  // There might be ABI-specific function attributes that are set by this method
  // that should be handled here. For example, attributes such as
  // "__attribute__((stdcall))" that enforce a specific calling convention.
  // However, most of these attributes are handled in CIRGen and there's
  // currently no implementation of such in CIR.

  // TODO(cir): Regarding CodeGenModule::GetCPUAndFeaturesAttributes:
  // This method sets the CPU and Features attributes for the function (e.g.
  // mmx, x87, etc.), which a target specific function attribute, but unrelated
  // to the ABI and call conv. I would prefer to handle this on a separate
  // target lowering pass.

  // Collect attributes from arguments and return values.
  CIRToCIRArgMapping IRFunctionArgs(getContext(), FI);

  // Type RetTy = FI.getReturnType();
  const ABIArgInfo &RetAI = FI.getReturnInfo();
  // const CIRDataLayout &DL = getDataLayout();

  // TODO(cir): NoUndef attribute for return values partially depends on
  // ABI-specific information. Maybe we should include it here.

  switch (RetAI.getKind()) {
  case ABIArgInfo::Extend:
    if (RetAI.isSignExt())
      newFn.setResultAttr(0, "cir.signext", rewriter.getUnitAttr());
    else
      // FIXME(cir): Add a proper abstraction to create attributes.
      newFn.setResultAttr(0, "cir.zeroext", rewriter.getUnitAttr());
    [[fallthrough]];
  case ABIArgInfo::Direct:
    if (RetAI.getInReg())
      llvm_unreachable("InReg attribute is NYI");
    assert(MissingFeature::noFPClassAttr());
    break;
  case ABIArgInfo::Ignore:
    break;

  default:
    llvm_unreachable("Missing ABIArgInfo::Kind");
  }

  if (!IsThunk) {
    if (!MissingFeature::isReferenceType()) {
      llvm_unreachable("Reference handling is NYI");
    }
  }

  // bool hasUsedSRet = false;
  // SmallVector<NamedAttrList, 4> ArgAttrs(IRFunctionArgs.totalIRArgs());

  // Attach attributes to sret.
  if (!MissingFeature::sretArgument()) {
    llvm_unreachable("sret is NYI");
  }

  // Attach attributes to inalloca argument.
  if (!MissingFeature::inallocaArgument()) {
    llvm_unreachable("inalloca is NYI");
  }

  // Apply `nonnull`, `dereferencable(N)` and `align N` to the `this` argument,
  // unless this is a thunk function.
  // FIXME: fix this properly, https://reviews.llvm.org/D100388
  if (!MissingFeature::isMethod() || !MissingFeature::inallocaArgument()) {
    llvm_unreachable("`this` argument attributes are NYI");
  }

  unsigned ArgNo = 0;
  for (LoweringFunctionInfo::const_arg_iterator I = FI.arg_begin(),
                                                E = FI.arg_end();
       I != E; ++I, ++ArgNo) {
    // Type ParamType = I->type;
    const ABIArgInfo &AI = I->info;
    SmallVector<NamedAttribute> Attrs;

    // Add attribute for padding argument, if necessary.
    if (IRFunctionArgs.hasPaddingArg(ArgNo)) {
      llvm_unreachable("Padding argument is NYI");
    }

    // TODO(cir): Mark noundef arguments and return values. Although this
    // attribute is not a part of the call conve, it uses it to determine if a
    // value is noundef (e.g. if an argument is passed direct, indirectly, etc).

    // 'restrict' -> 'noalias' is done in EmitFunctionProlog when we
    // have the corresponding parameter variable.  It doesn't make
    // sense to do it here because parameters are so messed up.
    switch (AI.getKind()) {
    case ABIArgInfo::Extend:
      if (AI.isSignExt())
        Attrs.push_back(
            rewriter.getNamedAttr("cir.signext", rewriter.getUnitAttr()));
      else
        // FIXME(cir): Add a proper abstraction to create attributes.
        Attrs.push_back(
            rewriter.getNamedAttr("cir.zeroext", rewriter.getUnitAttr()));
      [[fallthrough]];
    case ABIArgInfo::Direct:
      if (ArgNo == 0 && !MissingFeature::chainCall())
        llvm_unreachable("ChainCall is NYI");
      else if (AI.getInReg())
        llvm_unreachable("InReg attribute is NYI");
      // Attrs.addStackAlignmentAttr(llvm::MaybeAlign(AI.getDirectAlign()));
      assert(MissingFeature::noFPClassAttr());
      break;
    default:
      llvm_unreachable("Missing ABIArgInfo::Kind");
    }

    if (!MissingFeature::isReferenceType()) {
      llvm_unreachable("Reference handling is NYI");
    }

    // TODO(cir): Missing some swift and nocapture stuff here.
    assert(MissingFeature::extParamInfo());

    if (!Attrs.empty()) {
      unsigned FirstIRArg, NumIRArgs;
      std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);
      for (unsigned i = 0; i < NumIRArgs; i++)
        newFn.setArgAttrs(FirstIRArg + i, Attrs);
    }
  }
  assert(ArgNo == FI.arg_size());

  // NOTE(cir): We do not need to set the Attrs argument here. We are binding
  // the attributes to the function as we go.
}

void LoweringModule::setCIRFunctionAttributes(FuncOp GD,
                                              const LoweringFunctionInfo &Info,
                                              FuncOp F, bool IsThunk) {
  unsigned CallingConv;
  // NOTE(cir): The method below will update the F function with the proper
  // attributes.
  constructAttributeList(GD.getName(), Info, GD, F, CallingConv,
                         /*AttrOnCallSite=*/false, IsThunk);
  // TODO(cir): Set Function's calling convention.
}

void LoweringModule::setFunctionAttributes(FuncOp FD, FuncOp F,
                                           bool IsIncompleteFunction,
                                           bool IsThunk) {
  // TODO(cir): Add query in FuncOp to check if it is complete.
  // FIXME(cir): Why do we call arrageGlobalDeclaration here? We already have F
  // which is the function we want to lower.
  if (!IsIncompleteFunction)
    setCIRFunctionAttributes(FD, getTypes().arrangeGlobalDeclaration(FD), F,
                             IsThunk);

  // Add the Returned attribute for "this", except for iOS 5 and earlier
  // where substantial code, including the libstdc++ dylib, was compiled with
  // GCC and does not actually return "this".
  if (!IsThunk && getCXXABI().hasThisReturn(FD) &&
      !(getTriple().isiOS() && getTriple().isOSVersionLT(6))) {
    llvm_unreachable("Returned attribute is NYI");
  }

  // NOTE(cir): Linkage is handled in CIRGen. No need to set it here.
}

/// If the specified mangled name is not in the module, create and return an
/// llvm Function with the specified type. If there is something in the module
/// with the specified name, return it potentially bitcasted to the right
/// type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is
/// used to set the attributes on the function when it is first created.
FuncOp LoweringModule::getOrCreateCIRFunction(StringRef MangledName,
                                              FuncType Ty, FuncOp GD,
                                              bool ForVTable, bool DontDefer,
                                              bool IsThunk,
                                              ArrayRef<Attribute> ExtraAttrs,
                                              bool IsForDefinition) {

  // NOTE(cir): Skip some multi-version functio stuff here. This should be
  // handled in CIRGen.

  // NOTE(cir): Skip entry lookup and creation. In this pass we already have
  // the function we want to lower.

  // NOTE(cir): Also skip incomplete function's handling. CIRGen will take
  // that.

  // NOTE(cir): Here we clone the original function without regions allowing
  // us to preserve everything except the type and body, which will both be
  // replaced by an ABI-specific code. Attributes may be added, but the
  // existing ones will be preserved. If some attribute should be dropped or
  // converted to an ABI-specific one, do it here.
  FuncOp F = cast<FuncOp>(rewriter.cloneWithoutRegions(GD));
  F.setType(Ty);

  // NOTE(cir): Skip declaration annotation and some other no-proto handling.

  // Set up function attributes.
  setFunctionAttributes(GD, F, false, IsThunk);
  llvm::outs() << "New Attributes: \n"
               << "\tResult: " << F.getResAttrs() << "\n"
               << "\tArgs: " << F.getArgAttrs() << "\n";
  if (!ExtraAttrs.empty()) {
    llvm_unreachable("ExtraAttrs are NYI");
  }

  // FIXME(cir): Does this make sense here? Deferring is a codegen thing, I
  // think. There is som ABI-specific stuff here though.
  if (!DontDefer) {
    llvm_unreachable("DontDefer is NYI");
  } else if (!MissingFeature::langOptions()) {
    llvm_unreachable("LangOptions is NYI");
  }

  // NOTE(cir): Leave incomplete function checks to CIRGen.

  return F;
}

/// Return the address of the given function.  If Ty is non-null, then this
/// function will use the specified type if it has to create it (this occurs
/// when we see a definition of the function).
///
/// NOTE(cir): This is a partial copy of the original CodeGenModule's
/// GetAddrOfFunction method. A lot of codegen stuff is ignored here as it's
/// handled in CIRGen.
FuncOp LoweringModule::getAddrOfFunction(FuncOp GD, FuncType Ty, bool ForVTable,
                                         bool DontDefer, bool IsForDefinition) {
  // If there was no specific requested type, just convert it now.
  if (!Ty) {
    llvm_unreachable("Might happen in CIRGen, but not here, I think.");
  }

  // FIXME(cir): we might need to identify ctor and dtors for ABI lowering
  // here.
  if (!MissingFeature::isCtorOrDtor()) {
    llvm_unreachable("Ctors and Dtors are not supported");
  }

  // NOTE(cir): No need to get the mangled name here. CIR's highest level
  // already uses the mangled name (though we might change this later).
  auto F = getOrCreateCIRFunction(GD.getName(), Ty, GD, ForVTable, DontDefer);

  if (!MissingFeature::langOptions() && !MissingFeature::CUDA()) {
    llvm_unreachable("CUDA is NYI");
  }
  return F;
}

/// Rewrites an existing function to conform to the ABI (e.g. follow calling
/// conventions). This method tries to follow the original
/// CodeGenModule::EmitGlobalFunctionDefinition method as closely as possible.
/// However, they are inherently different.
void LoweringModule::rewriteGlobalFunctionDefinition(
    FuncOp op, LoweringModule &state, PatternRewriter &rewriter) {
  const LoweringFunctionInfo &FI =
      state.getTypes().arrangeGlobalDeclaration(op);
  FuncType Ty = state.getTypes().getFunctionType(FI);

  llvm::outs() << "Call Conv Lowering\n"
               << "\tfrom: " << op.getFunctionType() << "\n"
               << "\t  to: " << Ty << "\n";

  // NOTE(cir): Even if the old function type and the new function type are
  // the same, we still need to rewrite the function to conform to the ABI in
  // most cases. For example, `void (u32i)` will be lowered to `void (zeroext
  // u32i)`.

  // FIXME(cir): The clone below might be flawed. For example, if a parameter
  // has an attribute but said parameter is coerced to multiple parameters, we
  // will not perform any for of mapping or attribute drop to account for
  // this. We need a proper procedure to rewrite a FuncOp and its properties
  // properly.
  FuncOp newFn =
      getAddrOfFunction(op, Ty, /*ForVTable=*/false, /*DontDefer=*/true,
                        /*IsForDefinition=*/true);

  LowerFunction(*this, rewriter, op, newFn).generateCode(op, newFn, FI);

  // Erase original ABI-agnostic function.
  rewriter.eraseOp(op);
}

void LoweringModule::rewriteFunctionCall(CallOp op) {
  llvm::outs() << "Rewriting Call " << op.getCallee() << "\n";

  FuncOp callee = cast<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr()));

  LowerFunction(*this, rewriter, callee, op).rewriteCallOp(op);

  // Erase original ABI-agnostic call.
  rewriter.eraseOp(op);
}

} // namespace cir
} // namespace mlir
