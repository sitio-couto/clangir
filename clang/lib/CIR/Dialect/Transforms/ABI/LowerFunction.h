#pragma once

#include "CIRCXXABI.h"
#include "LoweringCall.h"
#include "LoweringFunctionInfo.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

/// The kind of evaluation to perform on values of a particular
/// type.  Basically, is the code in CGExprScalar, CGExprComplex, or
/// CGExprAgg?
///
/// TODO: should vectors maybe be split out into their own thing?
enum TypeEvaluationKind { TEK_Scalar, TEK_Complex, TEK_Aggregate };

class LowerFunction {
  LowerFunction(const LowerFunction &) = delete;
  void operator=(const LowerFunction &) = delete;

  friend class CIRCXXABI;

  const clang::TargetInfo &Target;

  PatternRewriter &rewriter;
  FuncOp SrcFn;  // Original ABI-agnostic function.
  FuncOp NewFn;  // New ABI-aware function.
  CallOp callOp; // Call operation to be lowered.

public:
  /// Builder for lowering calling convention of a function definition.
  LowerFunction(LoweringModule &cgm, PatternRewriter &rewriter, FuncOp srcFn,
                FuncOp newFn);

  /// Builder for lowering calling convention of a call operation.
  LowerFunction(LoweringModule &cgm, PatternRewriter &rewriter, FuncOp srcFn,
                CallOp callOp);

  ~LowerFunction() = default;

  LoweringModule &LM; // Per-module state.

  PatternRewriter &getRewriter() const { return rewriter; }

  const clang::TargetInfo &getTarget() const { return Target; }

  static bool hasAggregateEvaluationKind(Type T) {
    return getEvaluationKind(T) == TEK_Aggregate;
  }

  void emitFunctionProlog(const LoweringFunctionInfo &FI, FuncOp Fn,
                          MutableArrayRef<BlockArgument> Args);

  void emitFunctionEpilog(const LoweringFunctionInfo &FI);

  // TODO(cir): REVISE THIS CLASS.
  // It does not make much sense to have a class that follows codegen parity
  // considering that the ABI lowering pass is not codegen.

  // Parity with CodeGenFunction::StartFunction. Note that the Fn variable is
  // not a FuncOp, but a FuncType. In the original function, Fn is the result
  // LLVM IR function, but here we are going to .
  void startFunction(FuncOp GD, Type RetTy, FuncOp Fn,
                     llvm::MutableArrayRef<BlockArgument> &Args,
                     const LoweringFunctionInfo &FnInfo);

  /// Complete IR generation of the current function. It is legal to call this
  /// function even if there is no current insertion point.
  void finishFunction(const LoweringFunctionInfo &FI);

  // Parity with CodeGenFunction::GenerateCode. Keep in mind that several
  // sections in the original function are focused on codegen unrelated to the
  // ABI. Such sections are handled in CIR's codegen, not here.
  void generateCode(FuncOp GD, FuncOp Fn, const LoweringFunctionInfo &FnInfo);

  // Emit the most simple cir.store possible (e.g. a store for a whole
  // struct), which can later be broken down in other CIR levels (or prior
  // to dialect codegen).
  void buildAggregateStore(Value Val, Value Dest, bool DestIsVolatile);

  // Emit a trivial zero-extended store from a small integer value to an
  // allocated boolean value address.
  void buildBooleanStore(Value Val, Value Dest);

  // Emit a simple bitcast for a coerced aggregate type to convert it from an
  // ABI-agnostic to an ABI-aware type.
  Value buildAggregateBitcast(Value Val, Type DestTy);

  /// Rewrite a call operation to abide to the ABI calling convention.
  ///
  /// NOTE(cir): This method has partial parity to CodeGenFunction's
  /// EmitCallEpxr method. The core differences is that is focuses only on
  /// ABI-specific code emission and does not return a RValue.
  void rewriteCallOp(CallOp op, ReturnValueSlot retValSlot = ReturnValueSlot());

  /// Rewrite a call operation to abide to the ABI calling convention.
  ///
  /// NOTE(cir): This method has partial parity to CodeGenFunction's
  /// EmitCall method. A notable difference is that we also pass the call op
  /// which was already emitted in CIRGen.
  Value rewriteCallOp(FuncType calleeTy, FuncOp origCallee, CallOp callOp,
                      ReturnValueSlot retValSlot, Value Chain = nullptr);
  // FIXME(cir): We should make LoweringFunctionInfo carry the original
  // function/call which is being lowered.
  Value rewriteCallOp(const LoweringFunctionInfo &CallInfo, FuncOp Callee,
                      CallOp Caller, ReturnValueSlot ReturnValue,
                      SmallVector<Value> &CallArgs, CallOp CallOrInvoke,
                      bool isMustTail, Location loc);

  /// Return the TypeEvaluationKind of Type \c T.
  static TypeEvaluationKind getEvaluationKind(Type T);
};

} // namespace cir
} // namespace mlir
