#include "LowerFunction.h"
#include "CIRToCIRArgMapping.h"
#include "LoweringFunctionInfo.h"
#include "LoweringModule.h"
#include "MissingFeature.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

namespace {

Value buildAddressAtOffset(LowerFunction &LF, Value addr,
                           const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    llvm_unreachable("NYI");
  }
  return addr;
}

/// CreateCoercedStore - Create a store to \arg DstPtr from \arg Src,
/// where the source and destination may have different types.  The
/// destination is known to be aligned to \arg DstAlign bytes.
///
/// This safely handles the case when the src type is larger than the
/// destination type; the upper bits of the src will be lost.
void createCoercedStore(Value Src, Value Dst, bool DstIsVolatile,
                        LowerFunction &CGF) {
  llvm_unreachable("NYI");
}

} // namespace

LowerFunction::LowerFunction(LoweringModule &lm, PatternRewriter &rewriter)
    : LM(lm), Target(lm.getTarget()), rewriter(rewriter) {}

void LowerFunction::emitFunctionProlog(const LoweringFunctionInfo &FI,
                                       FuncOp Fn,
                                       MutableArrayRef<BlockArgument> Args) {
  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), FI);
  assert(Fn.getNumArguments() == IRFunctionArgs.totalIRArgs());

  assert(MissingFeature::inallocaArgument());
  assert(MissingFeature::sretArgument());

  // Track if we received the parameter as a pointer (indirect, byval, or
  // inalloca). If already have a pointer, EmitParmDecl doesn't need to copy it
  // into a local alloca for us.
  SmallVector<Value, 8> ArgVals;
  ArgVals.reserve(Args.size());

  // Create a pointer value for every parameter declaration.  This usually
  // entails copying one or more LLVM IR arguments into an alloca.  Don't push
  // any cleanups or do anything that might unwind.  We do that separately, so
  // we can push the cleanups in the correct order for the ABI.
  assert(FI.arg_size() == Args.size());

  unsigned ArgNo = 0;
  LoweringFunctionInfo::const_arg_iterator info_it = FI.arg_begin();
  for (MutableArrayRef<BlockArgument>::const_iterator i = Args.begin(),
                                                      e = Args.end();
       i != e; ++i, ++info_it, ++ArgNo) {
    const Value Arg = *i;
    const ABIArgInfo &ArgI = info_it->info;

    bool isPromoted = !MissingFeature::isKNRPromoted();
    // We are converting from ABIArgInfo type to VarDecl type directly, unless
    // the parameter is promoted. In this case we convert to
    // CGFunctionInfo::ArgInfo type with subsequent argument demotion.
    Type Ty = {};
    if (isPromoted)
      llvm_unreachable("NYI");
    else
      Ty = Arg.getType();
    assert(MissingFeature::evaluationKind());

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgI.getKind()) {
    case ABIArgInfo::Direct: {
      auto AI = Fn.getArgument(FirstIRArg);
      Type LTy = Arg.getType();

      // Prepare parameter attributes. So far, only attributes for pointer
      // parameters are prepared. See
      // http://llvm.org/docs/LangRef.html#paramattrs.
      if (ArgI.getDirectOffset() == 0 && LTy.isa<PointerType>() &&
          ArgI.getCoerceToType().isa<PointerType>()) {
        llvm_unreachable("NYI");
      }

      // Prepare the argument value. If we have the trivial case, handle it
      // with no muss and fuss.
      if (!isa<StructType>(ArgI.getCoerceToType()) &&
          ArgI.getCoerceToType() == Ty && ArgI.getDirectOffset() == 0) {
        llvm_unreachable("NYI");
      }

      assert(MissingFeature::vectorType());

      // Allocate original argument to be "uncoerced".
      // FIXME(cir): We should have a alloca op builder that does not required
      // the pointer type to be explicitly passed.
      // FIXME(cir): Get the original name of the argument, as well as the
      // proper alignment for the given type being allocated.
      auto Alloca = rewriter.create<AllocaOp>(
          Fn.getLoc(), rewriter.getType<PointerType>(Ty), Ty,
          /*name=*/StringRef(""),
          /*alignment=*/rewriter.getAttr<IntegerAttr>(APSInt(4)));

      Value Ptr = buildAddressAtOffset(*this, Alloca.getResult(), ArgI);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      StructType STy = dyn_cast<StructType>(ArgI.getCoerceToType());
      if (ArgI.isDirect() && ArgI.getCanBeFlattened() && STy &&
          STy.getNumElements() > 1) {
        llvm_unreachable("NYI");
      } else {
        // Simple case, just do a coerced store of the argument into the alloca.
        assert(NumIRArgs == 1);
        Value AI = Fn.getArgument(FirstIRArg);
        // TODO(cir): Set argument name in the new function.
        createCoercedStore(AI, Ptr, /*DstIsVolatile=*/false, *this);
      }

      // Match to what EmitParamDecl is expecting for this type.
      if (!MissingFeature::evaluationKind()) {
        llvm_unreachable("NYI");
      } else {
        // FIXME(cir): Should we have an ParamValue abstraction like in the
        // original codegen?
        ArgVals.push_back(Alloca);
      }
      break;
    }
    default:
      llvm_unreachable("NYI");
    }
  }

  if (getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
    llvm_unreachable("NYI");
  } else {
    for (unsigned I = 0, E = Args.size(); I != E; ++I) {
      // TODO(cir): This requires a codegen method. We can duplicated it, which
      // seems like an awful idea, or we create something new. Dunno.
      llvm_unreachable("NYI");
    }
  }
}

void LowerFunction::generateCode(FuncOp GD, FuncOp Fn,
                                 const LoweringFunctionInfo &FnInfo) {
  auto Args = GD.getArguments();
  Type ResTy = GD.getFunctionType().getReturnType();

  // NOTE(cir): Skipped some inline stuff from codegen here. Unlinkely that we
  // will need it for ABI lowering.

  // NOTE(cir): We may have to emit/edit function debug info here.

  startFunction(GD, ResTy, Fn, Args, FnInfo);
}

// Parity with CodeGenFunction::StartFunction. Note that the Fn variable is not
// a FuncOp, but a FuncType. In the original function, Fn is the result LLVM IR
// function, but here we are going to .
void LowerFunction::startFunction(FuncOp GD, Type RetTy, FuncOp Fn,
                                  llvm::MutableArrayRef<BlockArgument> &Args,
                                  const LoweringFunctionInfo &FnInfo) {
  // NOTE(cir): In the original Clang codegen, a lot of stuff is done here.
  // However, in CIR, we split this function between codegen and ABI lowering.
  // This means that the following sections are not necessary here as they will
  // be handled in CIR's codegen:
  // - Handling of sanitizers.
  // - Profiling.
  // - Addition/removal of function attributes.

  auto *entry = Fn.addEntryBlock();
  rewriter.setInsertionPointToEnd(entry);

  emitFunctionProlog(FnInfo, Fn, GD.getArguments());

  return;
}

} // namespace cir
} // namespace mlir
