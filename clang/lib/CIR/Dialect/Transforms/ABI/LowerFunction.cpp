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
#include "llvm/Support/TypeSize.h"
#include <algorithm>

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

/// Given a struct pointer that we are accessing some number of bytes out of it,
/// try to gep into the struct to get at its inner goodness.  Dive as deep as
/// possible without entering an element with an in-memory size smaller than
/// DstSize.
static Value enterStructPointerForCoercedAccess(Value SrcPtr, StructType SrcSTy,
                                                uint64_t DstSize,
                                                LowerFunction &CGF) {
  // We can't dive into a zero-element struct.
  if (SrcSTy.getNumElements() == 0)
    llvm_unreachable("NYI");

  Type FirstElt = SrcSTy.getMembers()[0];

  // If the first elt is at least as large as what we're looking for, or if the
  // first element is the same size as the whole struct, we can enter it. The
  // comparison must be made on the store size and not the alloca size. Using
  // the alloca size may overstate the size of the load.
  uint64_t FirstEltSize = CGF.LM.getDataLayout().getTypeStoreSize(FirstElt);
  if (FirstEltSize < DstSize &&
      FirstEltSize < CGF.LM.getDataLayout().getTypeStoreSize(SrcSTy))
    return SrcPtr;

  llvm_unreachable("NYI");
}

/// CreateCoercedStore - Create a store to \arg Dst from \arg Src,
/// where the source and destination may have different types.
///
/// This safely handles the case when the src type is larger than the
/// destination type; the upper bits of the src will be lost.
void createCoercedStore(Value Src, Value Dst, bool DstIsVolatile,
                        LowerFunction &CGF) {
  Type SrcTy = Src.getType();
  Type DstTy = Dst.getType();
  if (SrcTy == DstTy) {
    llvm_unreachable("NYI");
  }

  // NOTE(cir): In CIR, booleans are not a trivial coercion. Because of this
  // they are handled here.
  if (SrcTy.isa<IntType>() && DstTy.isa<PointerType>() &&
      DstTy.cast<PointerType>().getPointee().isa<BoolType>()) {
    CGF.buildBooleanStore(Src, Dst);
    return;
  }

  // FIXME(cir): We need a better way to handle datalayout queries.
  assert(SrcTy.isa<IntType>());
  llvm::TypeSize SrcSize = CGF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  if (StructType DstSTy = DstTy.dyn_cast<StructType>()) {
    Dst = enterStructPointerForCoercedAccess(Dst, DstSTy,
                                             SrcSize.getFixedValue(), CGF);
    assert(Dst.getType().isa<PointerType>());
    DstTy = Dst.getType().cast<PointerType>().getPointee();
  }

  PointerType SrcPtrTy = SrcTy.dyn_cast<PointerType>();
  PointerType DstPtrTy = DstTy.dyn_cast<PointerType>();
  // TODO(cir): Implement address space.
  if (SrcPtrTy && DstPtrTy && MissingFeature::addresSpace()) {
    llvm_unreachable("NYI");
  }

  // If the source and destination are integer or pointer types, just do an
  // extension or truncation to the desired type.
  if ((SrcTy.isa<IntegerType>() || SrcTy.isa<PointerType>()) &&
      (DstTy.isa<IntegerType>() || DstTy.isa<PointerType>())) {
    llvm_unreachable("NYI");
  }

  llvm::TypeSize DstSize = CGF.LM.getDataLayout().getTypeAllocSize(DstTy);

  // If store is legal, just bitcast the src pointer.
  assert(MissingFeature::vectorType());
  if (SrcSize.getFixedValue() <= DstSize.getFixedValue()) {
    // Dst = Dst.withElementType(SrcTy);
    CGF.buildAggregateStore(Src, Dst, DstIsVolatile);
  } else {
    llvm_unreachable("NYI");
  }
}

} // namespace

LowerFunction::LowerFunction(LoweringModule &lm, PatternRewriter &rewriter,
                             FuncOp srcFn)
    : Target(lm.getTarget()), rewriter(rewriter), SrcFn(srcFn), LM(lm) {}

/// This method has partial parity with CodeGenFunction::EmitFunctionProlog from
/// the original codegen. However, it focuses on the ABI-specific details. On
/// top of that, it is also responsible for rewriting the original function.
void LowerFunction::emitFunctionProlog(const LoweringFunctionInfo &FI,
                                       FuncOp Fn,
                                       MutableArrayRef<BlockArgument> Args) {
  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), FI);
  assert(Fn.getNumArguments() == IRFunctionArgs.totalIRArgs());

  // If we're using inalloca, all the memory arguments are GEPs off of the last
  // parameter, which is a pointer to the complete memory area.
  assert(MissingFeature::inallocaArgument());

  // Name the struct return parameter.
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
    case ABIArgInfo::Extend:
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
        assert(NumIRArgs == 1);

        // LLVM expects swifterror parameters to be used in very restricted
        // ways. Copy the value into a less-restricted temporary.
        Value V = AI;
        if (!MissingFeature::extParamInfo()) {
          llvm_unreachable("NYI");
        }

        // Ensure the argument is the correct type.
        if (V.getType() != ArgI.getCoerceToType())
          llvm_unreachable("NYI");

        if (isPromoted)
          llvm_unreachable("NYI");

        ArgVals.push_back(V);

        // NOTE(cir): Here we have a trivial case, which means we can just
        // replace all uses of the original argument with the new one.
        Value oldArg = SrcFn.getArgument(ArgNo);
        Value newArg = Fn.getArgument(FirstIRArg);
        rewriter.replaceAllUsesWith(oldArg, newArg);

        break;
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
          /*alignment=*/rewriter.getI64IntegerAttr(4));

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

      // NOTE(cir): Once we have uncoerced the argument, we should be able to
      // RAUW the original argument alloca with the new one. This assumes that
      // the argument is used only to be stored in a alloca.
      Value arg = SrcFn.getArgument(ArgNo);
      assert(arg.hasOneUse());
      for (auto *firstStore : arg.getUsers()) {
        assert(isa<StoreOp>(firstStore));
        auto argAlloca = cast<StoreOp>(firstStore).getAddr();
        rewriter.replaceAllUsesWith(argAlloca, Alloca);
        rewriter.eraseOp(firstStore);
        rewriter.eraseOp(argAlloca.getDefiningOp());
      }

      break;
    }
    default:
      llvm_unreachable("Unhandled ABIArgInfo::Kind");
    }
  }

  if (getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
    llvm_unreachable("NYI");
  } else {
    // FIXME(cir): This requires a codegen method. For the example I'm testing,
    // it does nothing, but we will likely need to duplicated it here in the
    // future. for (unsigned I = 0, E = Args.size(); I != E; ++I)
    //   buildParamDecl(Args[I], ArgVals[I], I + 1);
    llvm::errs() << "Skipping buildParamDecl in emitFunctionProlog\n";
  }
}

void LowerFunction::emitFunctionEpilog(const LoweringFunctionInfo &FI) {
  // NOTE(cir): no-return, naked, and no result functions should be handled in
  // CIRGen.

  Value RV = {};
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  // switch (RetAI.getKind()) {
  // default:
  //   llvm_unreachable("Unhandled ABIArgInfo::Kind");
  // }

  // if (RV) {
  //   llvm_unreachable("NYI");
  // } else {
  //   llvm_unreachable("NYI");
  // }
}

void LowerFunction::finishFunction(const LoweringFunctionInfo &FI) {
  // Emit the standard function epilogue.
  emitFunctionEpilog(FI);
}

void LowerFunction::generateCode(FuncOp GD, FuncOp Fn,
                                 const LoweringFunctionInfo &FnInfo) {
  auto Args = GD.getArguments();
  Type ResTy = GD.getFunctionType().getReturnType();

  // NOTE(cir): Skipped some inline stuff from codegen here. Unlinkely that we
  // will need it for ABI lowering.

  // NOTE(cir): We may have to emit/edit function debug info here. Skipping it
  // for now.

  // NOTE(cir): Let codegen handle location stuff. No need to do it here.

  // NOTE(cir): Lifetime markers should be dealt with in codegen.

  // Emit the ABI-specific function prologue.
  startFunction(GD, ResTy, Fn, Args, FnInfo);

  // NOTE(cir): Now that we re-emitted the function with an ABI-specific
  // prologue, we have to migrate the function's body. This assumes that all
  // arguments of the original function were RAUW'd with the new ones.
  // FIXME(cir): The implementation below is pretty trashy: will not work if
  // SrcFn has multiple blocks; mixes the new and old prologues.
  // FIXME(cir): Perhaps we can leverage MLIR's SignatureConversion to do this.
  assert(std::all_of(Args.begin(), Args.end(),
                     [](auto arg) { return arg.getUses().empty(); }) &&
         "Missing RAUW?");
  assert(SrcFn.getBody().hasOneBlock() &&
         "Multiple blocks in original function not supported");
  rewriter.mergeBlocks(&SrcFn.getBody().front(), &Fn.getBody().front(),
                       Fn.getArguments());

  // FIXME(cir): What about saving parameters for corotines? Should we do
  // something about it in this pass? If the change with the calling convention,
  // we might have to handle this here.

  // NOTE(cir): In the original codegen, this is where the function's body is
  // generated. Since we already did this, and this pass lower's only calling
  // conventions, we don't need to do anything here.

  // TODO(cir): We should handle return values here as well.

  // Emit the standard function epilogue.
  finishFunction(FnInfo);
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

void LowerFunction::buildAggregateStore(Value Val, Value Dest,
                                        bool DestIsVolatile) {
  // In LLVM codegen:
  // Function to store a first-class aggregate into memory. We prefer to
  // store the elements rather than the aggregate to be more friendly to
  // fast-isel.
  assert(Dest.getType().isa<PointerType>() && "Storing in a non-pointer!");
  (void)DestIsVolatile;

  // Circumvent CIR's type checking.
  Type pointeeTy = Dest.getType().cast<PointerType>().getPointee();
  if (Val.getType() != pointeeTy) {
    // NOTE(cir):  We only bitcast and store if the types have the same size.
    assert((LM.getDataLayout().getTypeSizeInBits(Val.getType()) ==
            LM.getDataLayout().getTypeSizeInBits(pointeeTy)) &&
           "Incompatible types");
    auto loc = Val.getLoc();
    Val = rewriter.create<CastOp>(loc, pointeeTy, CastKind::bitcast, Val);
  }

  rewriter.create<StoreOp>(Val.getLoc(), Val, Dest);
}

void LowerFunction::buildBooleanStore(Value Val, Value Dest) {
  assert(Val.getType().isa<IntType>() && "Not an integer type");
  assert(Dest.getType().isa<PointerType>() && "Storing in a non-pointer!");

  const auto loc = Val.getLoc();
  Type pointeeTy = Dest.getType().cast<PointerType>().getPointee();

  Val = rewriter.create<CastOp>(loc, pointeeTy, CastKind::bitcast, Val);
  rewriter.create<StoreOp>(loc, Val, Dest);
}

/// Emit an alloca (or GlobalValue depending on target)
/// for the specified parameter and set up LocalDeclMap.
void LowerFunction::buildParmDecl(const BlockArgument D, Value Arg,
                                  unsigned ArgNo) {
  // bool NoDebugInfo = false;

  // // TODO(cir): When will Arg be a global?
  // // TODO(cir): Set the name of the alloca.
  // assert(!isa<GetGlobalOp>(Arg.getDefiningOp()));

  // Type Ty = D.getType();

  // // Use better IR generation for certain implicit parameters.
  // if (MissingFeature::implicitParamDecl()) {
  //   llvm_unreachable("NYI");
  // }

  // Value DeclPtr = {};
  // Value AllocaPtr = {};
  // bool DoStore = false;
  // assert(MissingFeature::evaluationKind());
  // bool UseIndirectDebugAddress = false;

  // // If we already have a pointer to the argument, reuse the input pointer.
  // if (Arg.isIndirect()) {
  //   DeclPtr = Arg.getIndirectAddress();
  //   DeclPtr = DeclPtr.withElementType(Ty);

  //   // TODO(cir): needs arg info here.
  // } else {
  //   llvm_unreachable("NYI");
  // }
}

} // namespace cir
} // namespace mlir
