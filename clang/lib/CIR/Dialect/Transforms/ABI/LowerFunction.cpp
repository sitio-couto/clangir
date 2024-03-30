#include "LowerFunction.h"
#include "CIRToCIRArgMapping.h"
#include "LoweringCall.h"
#include "LoweringFunctionInfo.h"
#include "LoweringModule.h"
#include "MissingFeature.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/STLExtras.h"
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

/// Create a store to \param Dst from \param Src where the source and
/// destination may have different types.
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

/// Create a load from \param SrcPtr interpreted as  a pointer to an object of
/// type \param Ty, known to be aligned to  \param SrcAlign bytes.
///
/// This safely handles the case when the src type is smaller than the
/// destination type; in this situation the values of bits which not present in
/// the src are undefined.
///
/// NOTE(cir): This method has partial parity with CGCall's CreateCoercedLoad.
/// Unlike the original codegen, this function does not emit a coerced load
/// since CIR's type checker wouldn't allow it. Instead, it casts the existing
/// ABI-agnostic value to it's ABI-aware counterpart. Nevertheless, we should
/// try to follow the same logic as the original codegen for correctness.
Value createCoercedValue(Value Src, Type Ty, LowerFunction &CGF) {
  Type SrcTy = Src.getType();

  // If SrcTy and Ty are the same, just reuse the exising load.
  if (SrcTy == Ty)
    return Src;

  llvm::TypeSize DstSize = CGF.LM.getDataLayout().getTypeAllocSize(Ty);

  if (auto SrcSTy = dyn_cast<StructType>(SrcTy)) {
    Src = enterStructPointerForCoercedAccess(Src, SrcSTy,
                                             DstSize.getFixedValue(), CGF);
    SrcTy = Src.getType();
  }

  llvm::TypeSize SrcSize = CGF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  // If the source and destination are integer or pointer types, just do an
  // extension or truncation to the desired type.
  if ((isa<IntType>(Ty) || isa<PointerType>(Ty)) &&
      (isa<IntType>(SrcTy) || isa<PointerType>(SrcTy))) {
    llvm_unreachable("NYI");
  }

  // If load is legal, just bitcast the src pointer.
  if (!SrcSize.isScalable() && !DstSize.isScalable() &&
      SrcSize.getFixedValue() >= DstSize.getFixedValue()) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    // Src = Src.withElementType();
    return CGF.buildAggregateBitcast(Src, Ty);
  }

  llvm_unreachable("NYI");
}

Value emitAddressAtOffset(LowerFunction &LF, Value addr,
                          const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    llvm_unreachable("NYI");
  }
  return addr;
}

/// After the calling convention is lowered, an ABI-agnostic type might have to
/// be loaded back to its ABI-aware couterpart so it may be returned. If they
/// differ, we have to do a coerced load. A coerced load, which means to load a
/// type to another despite that they represent the same value. The simplest
/// cases can be solved with a mere bitcast.
///
/// This partially replaces CreateCoercedLoad from the original codegen.
/// However, instead of emitting the load, it emits a cast.
///
/// FIXME(cir): Improve parity with the original codegen.
Value castReturnValue(Value Src, Type Ty, LowerFunction &LF) {
  Type SrcTy = Src.getType();

  // If SrcTy and Ty are the same, nothing to do.
  if (SrcTy == Ty)
    return Src;

  // If is the special boolean case, simply bitcast it.
  if (SrcTy.isa<BoolType>() && Ty.isa<IntType>()) {
    return LF.getRewriter().create<CastOp>(Src.getLoc(), Ty, CastKind::bitcast,
                                           Src);
  }

  llvm::TypeSize DstSize = LF.LM.getDataLayout().getTypeAllocSize(Ty);

  // FIXME(cir): Do we need the EnterStructPointerForCoercedAccess routine here?

  llvm::TypeSize SrcSize = LF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  if ((isa<IntType>(Ty) || isa<PointerType>(Ty)) &&
      (isa<IntType>(SrcTy) || isa<PointerType>(SrcTy))) {
    llvm_unreachable("NYI");
  }

  // If load is legal, just bitcast the src pointer.
  if (!SrcSize.isScalable() && !DstSize.isScalable() &&
      SrcSize.getFixedValue() >= DstSize.getFixedValue()) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    return LF.getRewriter().create<CastOp>(Src.getLoc(), Ty, CastKind::bitcast,
                                           Src);
  }

  llvm_unreachable("NYI");
}

/// Retrieve the alloca that stores the result of a call.
AllocaOp getResultAlloca(CallOp callOp) {
  auto result = callOp.getResult(0);
  assert(result.hasOneUse() && isa<StoreOp>(result.use_begin().getUser()));
  auto storeOp = cast<StoreOp>(result.use_begin().getUser());
  return storeOp.getAddr().getDefiningOp<AllocaOp>();
}

} // namespace

// FIXME(cir): Pass SrcFn and NewFn around instead of having then as attributes.
LowerFunction::LowerFunction(LoweringModule &lm, PatternRewriter &rewriter,
                             FuncOp srcFn, FuncOp newFn)
    : Target(lm.getTarget()), rewriter(rewriter), SrcFn(srcFn), NewFn(newFn),
      LM(lm) {}

LowerFunction::LowerFunction(LoweringModule &lm, PatternRewriter &rewriter,
                             FuncOp srcFn, CallOp callOp)
    : Target(lm.getTarget()), rewriter(rewriter), SrcFn(srcFn), callOp(callOp),
      LM(lm) {}

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
  Type RetTy = FI.getReturnType();
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  switch (RetAI.getKind()) {

  case ABIArgInfo::Ignore:
    break;

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    // FIXME(cir): Should we call ConvertType(RetTy) here?
    if (RetAI.getCoerceToType() == RetTy && RetAI.getDirectOffset() == 0) {
      // The internal return value temp always will have pointer-to-return-type
      // type, just do a load.

      // If there is a dominating store to ReturnValue, we can elide
      // the load, zap the store, and usually zap the alloca.
      // NOTE(cir): This seems like a premature optimization case, so I'm
      // skipping it.
      if (/*findDominatingStoreToReturnValue(*this)=*/false) {
        llvm_unreachable("NYI");
      }
      // Otherwise, we have to do a simple load.
      else {
        // NOTE(cir): Nothing to do here. The codegen already emitted this load
        // for us and there is no casting necessary to conform to the ABI. The
        // zero-extension is enforced by the return value's attribute. Just
        // early exit.
        return;
      }
    } else {
      // NOTE(cir): Unlike the original codegen, CIR may have multiple return
      // statements in the function body. We have to handle this here.
      mlir::PatternRewriter::InsertionGuard guard(rewriter);
      NewFn->walk([&](ReturnOp returnOp) {
        rewriter.setInsertionPoint(returnOp);

        // NOTE(cir): I'm not sure if we need this offset here or in CIRGen.
        // Perhaps both? For now I'm just ignoring it.
        // Value V = emitAddressAtOffset(*this, getResultAlloca(returnOp),
        // RetAI);

        RV = castReturnValue(returnOp->getOperand(0), RetAI.getCoerceToType(),
                             *this);
        rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, RV);
      });
    }

    // TODO(cir): Should AutoreleaseResult be handled here?

    break;
  default:
    llvm_unreachable("Unhandled ABIArgInfo::Kind");
  }

  // NOTE(cir): Skip the creation of the return statement. We instead patch the
  // existing return statements with the new ABI-specific value.
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
  // FIXME(cir): Perhaps we can leverage MLIR's SignatureConversion to do
  // this.
  assert(std::all_of(Args.begin(), Args.end(),
                     [](auto arg) { return arg.getUses().empty(); }) &&
         "Missing RAUW?");
  assert(SrcFn.getBody().hasOneBlock() &&
         "Multiple blocks in original function not supported");

  // Move old function body to new function.
  rewriter.mergeBlocks(&SrcFn.getBody().front(), &Fn.getBody().front(),
                       Fn.getArguments());

  // FIXME(cir): What about saving parameters for corotines? Should we do
  // something about it in this pass? If the change with the calling
  // convention, we might have to handle this here.

  // NOTE(cir): In the original codegen, this is where the function's body is
  // generated. Since we already did this, and this pass lower's only calling
  // conventions, we don't need to do anything here.

  // TODO(cir): We should handle return values here as well.

  // Emit the standard function epilogue.
  finishFunction(FnInfo);
}

// Parity with CodeGenFunction::StartFunction. Note that the Fn variable is
// not a FuncOp, but a FuncType. In the original function, Fn is the result
// LLVM IR function, but here we are going to .
void LowerFunction::startFunction(FuncOp GD, Type RetTy, FuncOp Fn,
                                  llvm::MutableArrayRef<BlockArgument> &Args,
                                  const LoweringFunctionInfo &FnInfo) {
  // NOTE(cir): In the original Clang codegen, a lot of stuff is done here.
  // However, in CIR, we split this function between codegen and ABI lowering.
  // This means that the following sections are not necessary here as they
  // will be handled in CIR's codegen:
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

Value LowerFunction::buildAggregateBitcast(Value Val, Type DestTy) {
  return rewriter.create<CastOp>(Val.getLoc(), DestTy, CastKind::bitcast, Val);
}

void LowerFunction::buildBooleanStore(Value Val, Value Dest) {
  assert(Val.getType().isa<IntType>() && "Not an integer type");
  assert(Dest.getType().isa<PointerType>() && "Storing in a non-pointer!");

  const auto loc = Val.getLoc();
  Type pointeeTy = Dest.getType().cast<PointerType>().getPointee();

  Val = rewriter.create<CastOp>(loc, pointeeTy, CastKind::bitcast, Val);
  rewriter.create<StoreOp>(loc, Val, Dest);
}

/// Rewrite a call operation to abide to the ABI calling convention.
///
/// NOTE(cir): This method has partial parity to CodeGenFunction's
/// EmitCallEpxr method. The core differences is that is focuses only on
/// ABI-specific code emission and does not return a RValue.
void LowerFunction::rewriteCallOp(CallOp op, ReturnValueSlot retValSlot) {

  // TODO(cir): Check if BlockCall, CXXMemberCall, CUDAKernelCall,
  // CXXOperatorMember,  required special handling here. These should be handled
  // in CIRGen. If there is call conv or ABI-specific stuff to be handled, them
  // we should do it here.

  assert(SrcFn && "No source function");

  // TODO(cir): Also check if Builtin and CXXPeseudoDtor need special handling
  // here. These should be handled in CIRGen. If there is call conv or
  // ABI-specific stuff to be handled, them we should do it here.

  // NOTE(cir): There is no direct way to fetch the function type from the
  // CallOp, so we fecch it from the source function. The issue is that there is
  // no way to know if said type has already been ABI-lowered.
  rewriteCallOp(SrcFn.getFunctionType(), SrcFn, op, retValSlot);
}

/// Rewrite a call operation to abide to the ABI calling convention.
///
/// NOTE(cir): This method has partial parity to CodeGenFunction's
/// EmitCall method.
Value LowerFunction::rewriteCallOp(FuncType calleeTy, FuncOp origCallee,
                                   CallOp callOp, ReturnValueSlot retValSlot,
                                   Value Chain) {
  // NOTE(cir): Skip a bunch of function pointer stuff and AST declaration
  // asserts. Also skip sanitizers, as these should likely be handled at CIRGen.

  SmallVector<Value> Args;
  if (Chain)
    llvm_unreachable("NYI");

  // TODO(cir): Must identify CXX operator function calls.
  EvaluationOrder order = EvaluationOrder::Default;
  if (!MissingFeature::isCXXOperatorCall())
    llvm_unreachable("NYI");

  // NOTE(cir): Call args were already emitted in CIRGen. Just fetch them here.
  Args = callOp.getArgOperands();

  const LoweringFunctionInfo &FnInfo = LM.getTypes().arrangeFreeFunctionCall(
      callOp.getArgOperands(), calleeTy, /*chainCall=*/false);

  // C99 6.5.2.2p6:
  //   If the expression that denotes the called function has a type
  //   that does not include a prototype, [the default argument
  //   promotions are performed]. If the number of arguments does not
  //   equal the number of parameters, the behavior is undefined. If
  //   the function is defined with a type that includes a prototype,
  //   and either the prototype ends with an ellipsis (, ...) or the
  //   types of the arguments after promotion are not compatible with
  //   the types of the parameters, the behavior is undefined. If the
  //   function is defined with a type that does not include a
  //   prototype, and the types of the arguments after promotion are
  //   not compatible with those of the parameters after promotion,
  //   the behavior is undefined [except in some trivial cases].
  // That is, in the general case, we should assume that a call
  // through an unprototyped function type works like a *non-variadic*
  // call.  The way we make this work is to cast to the exact type
  // of the promoted arguments.
  //
  // Chain calls use this same code path to add the invisible chain parameter
  // to the function type.
  if (origCallee.getNoProto() || Chain) {
    llvm_unreachable("NYI");
  }

  assert(MissingFeature::CUDA());

  // TODO(cir): LLVM IR has the concept of "CallBase", which is a base class for
  // all types of calls. Perhaps we should have a CIR interface to mimic this
  // class.
  CallOp CallOrInvoke = {};
  Value Call =
      rewriteCallOp(FnInfo, origCallee, callOp, retValSlot, Args, CallOrInvoke,
                    /*isMustTail=*/false, callOp.getLoc());

  // NOTE(cir): Skipping debug stuff here.

  return Call;
}

Value LowerFunction::rewriteCallOp(const LoweringFunctionInfo &CallInfo,
                                   FuncOp Callee, CallOp Caller,
                                   ReturnValueSlot ReturnValue,
                                   SmallVector<Value> &CallArgs,
                                   CallOp CallOrInvoke, bool isMustTail,
                                   Location loc) {
  // FIXME: We no longer need the types from CallArgs; lift up and simplify.

  // Handle struct-return functions by passing a pointer to the
  // location that we would like to return into.
  Type RetTy = CallInfo.getReturnType();
  const ABIArgInfo &RetAI = CallInfo.getReturnInfo();

  FuncType IRFuncTy = LM.getTypes().getFunctionType(CallInfo);

  // NOTE(cir): Some target/ABI related checks happen here. I'm skipping them
  // under the assumption that they are handled in CIRGen.

  // 1. Set up the arguments.

  // If we're using inalloca, insert the allocation after the stack save.
  // FIXME: Do this earlier rather than hacking it in here!
  Value ArgMemory = {};
  if (StructType ArgStruct = CallInfo.getArgStruct()) {
    llvm_unreachable("NYI");
  }

  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), CallInfo);
  SmallVector<Value, 16> IRCallArgs(IRFunctionArgs.totalIRArgs());

  // If the call returns a temporary with struct return, create a temporary
  // alloca to hold the result, unless one is given to us.
  if (RetAI.isIndirect() || RetAI.isCoerceAndExpand() || RetAI.isInAlloca()) {
    llvm_unreachable("NYI");
  }

  assert(MissingFeature::Swift());

  // FIXME(cir): Do we need to track lifetime markers here?

  // Translate all of the arguments as necessary to match the IR lowering.
  assert(CallInfo.arg_size() == CallArgs.size() &&
         "Mismatch between function signature & arguments.");
  unsigned ArgNo = 0;
  LoweringFunctionInfo::const_arg_iterator info_it = CallInfo.arg_begin();
  for (auto I = CallArgs.begin(), E = CallArgs.end(); I != E;
       ++I, ++info_it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = info_it->info;

    if (IRFunctionArgs.hasPaddingArg(ArgNo))
      llvm_unreachable("NYI");

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Direct: {
      if (!isa<StructType>(ArgInfo.getCoerceToType()) &&
          ArgInfo.getCoerceToType() == info_it->type &&
          ArgInfo.getDirectOffset() == 0) {
        llvm_unreachable("NYI");
      }

      // FIXME: Avoid the conversion through memory if possible.
      Value Src = {};
      if (!I->getType().isa<StructType>()) {
        llvm_unreachable("NYI");
      } else {
        // NOTE(cir): I'm leaving L/RValue stuff for CIRGen to handle.
        Src = *I;
      }

      // If the value is offst in memory, apply the offset now.
      // FIXME(cir): Is this offset already handled in CIRGen?
      Src = emitAddressAtOffset(*this, Src, ArgInfo);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      StructType STy = dyn_cast<StructType>(ArgInfo.getCoerceToType());
      if (STy && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        llvm_unreachable("NYI");
      } else {
        // In the simple case, just pass the coerced loaded value.
        assert(NumIRArgs == 1);
        Value Load = createCoercedValue(Src, ArgInfo.getCoerceToType(), *this);

        // FIXME(cir): We should probably handle CMSE non-secure calls here

        // since they are a ARM-specific feature.
        if (!MissingFeature::argUndefAttr())
          llvm_unreachable("NYI");
        IRCallArgs[FirstIRArg] = Load;
      }

      break;
    }
    default:
      llvm::outs() << "Missing ABIArgInfo::Kind: " << ArgInfo.getKind() << "\n";
      llvm_unreachable("NYI");
    }

    // NOTE(cir): We don't need the callee func ptr here.

    if (ArgMemory || !MissingFeature::inallocaArgument()) {
      llvm_unreachable("NYI");
    }

    // NOTE(cir): There are some variadic related procedures here.
    if (Callee.getFunctionType().isVarArg()) {
      llvm_unreachable("NYI");
    }
  }

  // 3. Perform the actual call.

  // NOTE(cir): CIRGen handle when to "deactive" cleanups. We also skip some
  // debugging stuff here.

  // Update the largest vector width if any arguments have vector types.
  assert(MissingFeature::vectorType());

  // Compute the calling convention and attributes.

  // FIXME(cir): Skipping call attributes for now. Not sure if we have to do
  // this at all since we already do it for the function definition.

  // FIXME(cir): Implement the required procedures for strictfp function and
  // fast-math.

  // FIXME(cir): Add missing call-site attributes here if they are
  // ABI/target-specific, otherwise, do it in CIRGen.

  // NOTE(cir): Deciding whether to use Call or Invoke is done in CIRGen.

  // Rewrite the actual call operation.
  // TODO(cir): Handle other types of CIR calls (e.g. cir.try_call).
  // NOTE(cir): We don't know if the callee was already lowered, so we only
  // fetch the name from the callee, while the return type is fetch from the
  // lowering types manager.
  CallOp CI = rewriter.create<CallOp>(loc, Caller.getCalleeAttr(),
                                      IRFuncTy.getReturnType(), IRCallArgs);

  assert(MissingFeature::vectorType());

  // NOTE(cir): There some ObjC, tail-call, debug, and attribute stuff here that
  // I'm skipping.

  // 4. Finish the call.

  // If the call doesn't return, there is no need to translate the ABI-agnostic
  // return value to its ABI-aware counterpart.
  if (CI->getNumResults() == 0) {
    llvm_unreachable("NYI");
  }

  // NOTE(cir): Skipping some tail-call, swift, writeback, memory management
  // stuff here.

  // Extract the return value.
  Value Ret = [&] {
    switch (RetAI.getKind()) {
    case ABIArgInfo::Direct: {
      Type RetIRTy = RetTy;
      if (RetAI.getCoerceToType() == RetIRTy && RetAI.getDirectOffset() == 0) {
        llvm_unreachable("NYI");
      }

      // If coercing a fixed vector from a scalable vector for ABI
      // compatibility, and the types match, use the llvm.vector.extract
      // intrinsic to perform the conversion.
      if (!MissingFeature::vectorType()) {
        llvm_unreachable("NYI");
      }

      // FIXME(cir): Use return value slot here.
      Value RetVal = callOp.getResult(0);
      bool DestIsVolatile = ReturnValue.isVolatile();

      // NOTE(cir): If the function returns, there should always be a valid
      // return value present. Instead of setting the return value here, we
      // should have the ReturnValueSlot object set it beforehand.
      if (!RetVal) {
        RetVal = callOp.getResult(0);
        DestIsVolatile = false;
      }

      // An empty record can overlap other data (if declared with
      // no_unique_address); omit the store for such types - as there is no
      // actual data to store.
      if (RetTy.dyn_cast<StructType>() &&
          RetTy.cast<StructType>().getNumElements() != 0) {
        // NOTE(cir): I'm assuming we don't need to change any offsets here.
        // Value StorePtr = emitAddressAtOffset(*this, RetVal, RetAI);
        createCoercedValue(CI.getResult(0), RetVal.getType(), *this);
      }

      // NOTE(cir): No need to convert from a temp to an RValue. This is
      // done in CIRGen
      return RetVal;
    }
    default:
      llvm_unreachable("Unhandled ABIArgInfo::Kind");
    }
  }(); // FIXME(cir): Why does the original codegen does this weird
       // lambda thing?

  // NOTE(cir): Emissions, lifetime markers, and dtors are handled in CIRGen.

  return Ret;
}

/// Rewrite a call operation arguments to abide to the ABI calling convention.
///
/// NOTE(cir): This method has partial parity to CodeGenFunction's
/// EmitCallArgs method.
void LowerFunction::rewriteCallArgs(SmallVector<Value> &args, FuncType fnTy,
                                    OperandRange argRange, FuncOp callee,
                                    int paramsToSkip, EvaluationOrder order) {
  SmallVector<Type, 16> argTypes;

  assert((paramsToSkip == 0 || !callee.getNoProto()) &&
         "Can't skip parameters if type info is not provided");

  // This variable only captures *explicitly* written conventions, not those
  // applied by default via command line flags or target defaults, such as
  // thiscall, aapcs, stdcall via -mrtd, etc. Computing that correctly would
  // require knowing if this is a C++ instance method or being able to see
  // unprototyped FunctionTypes.
  clang::CallingConv ExplicitCC = clang::CallingConv::CC_C;

  // First, if a prototype was provided, use those argument types.
  bool IsVariadic = false;
  if (!callee.getNoProto()) {
    if (!MissingFeature::ObjC()) {
      llvm_unreachable("NYI");
    } else {
      IsVariadic = fnTy.isVarArg();
      // TODO(cir): Override calling convention if necessary.
      assert(MissingFeature::extParamInfo());
      argTypes.assign(fnTy.getInputs().begin() + paramsToSkip,
                      fnTy.getInputs().end());
    }
    // NOTE(cir): Skipping some debugging stuff here.
  }

  for (Value A : llvm::drop_begin(argRange, argTypes.size()))
    llvm_unreachable("NYI");
  assert((int)argTypes.size() == (argRange.end() - argRange.begin()));

  // We must evaluate arguments from right to left in the MS C++ ABI,
  // because arguments are destroyed left to right in the callee. As a special
  // case, there are certain language constructs that require left-to-right
  // evaluation, and in those cases we consider the evaluation order
  // requirement to trump the "destruction order is reverse construction
  // order" guarantee.
  bool LeftToRight =
      LM.getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()
          ? order == EvaluationOrder::ForceLeftToRight
          : order != EvaluationOrder::ForceRightToLeft;

  if (!MissingFeature::inallocaArgument()) {
    llvm_unreachable("NYI");
  }

  // Evaluate each argument in the appropriate order.
  size_t CallArgsStart = args.size();
  for (unsigned I = 0, E = argTypes.size(); I != E; ++I) {
    unsigned Idx = LeftToRight ? I : E - I - 1;
    auto Arg = argRange.begin() + Idx;
    unsigned InitialArgSize = args.size();
    // If *Arg is an ObjCIndirectCopyRestoreExpr, check that either the types
    // of the argument and parameter match or the objc method is
    // parameterized.
    assert(MissingFeature::ObjC());
    rewriteCallArg(args, *Arg, argTypes[Idx]);
    // In particular, we depend on it being the last arg in Args, and the
    // objectsize bits depend on there only being one arg if !LeftToRight.
    assert(InitialArgSize + 1 == args.size() &&
           "The code below depends on only adding one arg per EmitCallArg");
    (void)InitialArgSize;
    // Since pointer argument are never emitted as LValue, it is safe to emit
    // non-null argument check for r-value only.
    // FIXME(cir): Should we handle this non-null arg check here?
  }

  if (!LeftToRight) {
    llvm_unreachable("NYI");
  }
}

Value LowerFunction::rewriteAggExpr(Value aggregate) {
  llvm_unreachable("NYI");
}

Value LowerFunction::rewriteAnyExpr(Value V, bool ignoreResult) {
  switch (getEvaluationKind(V.getType())) {
  case TEK_Aggregate:
    // NOTE(cir): AggTemp creation is ignored here. We're rewriting stuff.
    // Creation is in CIRGen.
    return rewriteAggExpr(V);
  default:
    llvm::outs() << "Unhandled call argument kind: "
                 << getEvaluationKind(V.getType()) << "\n";
    llvm_unreachable("NYI");
  }
}

Value LowerFunction::rewriteAnyExprToTemp(Value V) {
  // NOTE(cir): I'm skipping AggValueSlot here for simplicity, but we might
  // need to handle that later. Also skipping the aggregate temp creation, as
  // it is already done in CIRGen.
  return rewriteAnyExpr(V);
}

void LowerFunction::rewriteCallArg(SmallVector<Value> &args, Value arg,
                                   Type argTy) {
  // NOTE(cir): Ignoring debugging info here. Handle it in CIRGen.
  if (!MissingFeature::ObjC()) {
    llvm_unreachable("NYI");
  }

  if (!MissingFeature::isGLValue()) {
    llvm_unreachable("NYI");
  }

  bool hasAggregateEvalKind = hasAggregateEvaluationKind(argTy);

  if (!MissingFeature::MSABI()) {
    llvm_unreachable("NYI");
  }

  // FIXME(cir): Skipping some L to RValue cast stuff here. Not sure how to
  // handle it.

  args.push_back(rewriteAnyExprToTemp(arg));
}

TypeEvaluationKind LowerFunction::getEvaluationKind(Type type) {
  // FIXME(cir): Implement type classes for CIR types.
  if (type.isa<StructType>())
    return TypeEvaluationKind::TEK_Aggregate;

  llvm_unreachable("NYI");
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
