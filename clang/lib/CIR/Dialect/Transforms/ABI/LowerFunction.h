#pragma once

#include "CIRCXXABI.h"
#include "LoweringFunctionInfo.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

class LowerFunction {
  LowerFunction(const LowerFunction &) = delete;
  void operator=(const LowerFunction &) = delete;

  friend class CIRCXXABI;

  const clang::TargetInfo &Target;

  PatternRewriter &rewriter;
  FuncOp SrcFn; // Original ABI-agnostic function.
  FuncOp NewFn; // New ABI-aware function.

public:
  LowerFunction(LoweringModule &cgm, PatternRewriter &rewriter, FuncOp srcFn,
                FuncOp newFn);
  ~LowerFunction() = default;

  LoweringModule &LM; // Per-module state.

  const clang::TargetInfo &getTarget() const { return Target; }

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

  //===--------------------------------------------------------------------===//
  //                            Declaration Emission
  //===--------------------------------------------------------------------===//

  // class ParamValue {
  //   Value value;
  //   Type type;
  //   unsigned Alignment;
  //   ParamValue(Value V, Type T, unsigned A)
  //       : value(V), type(T), Alignment(A) {}
  // public:
  //   static ParamValue forDirect(Value value) {
  //     return ParamValue(value, nullptr, 0);
  //   }
  //   static ParamValue forIndirect(Value addr) {
  //     assert(addr.getType().isa<PointerType>());
  //     addr.getType().cast<PointerType>().getABIAlignment(const
  //     ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef
  //     params); assert(!addr.getAlig().isZero()); return
  //     ParamValue(addr.getPointer(), addr.getElementType(),
  //                       addr.getAlignment().getQuantity());
  //   }

  //   bool isIndirect() const { return Alignment != 0; }
  //   Value *getAnyValue() const { return Value; }

  //   Value *getDirectValue() const {
  //     assert(!isIndirect());
  //     return Value;
  //   }

  //   Address getIndirectAddress() const {
  //     assert(isIndirect());
  //     return Address(Value, ElementType,
  //     CharUnits::fromQuantity(Alignment),
  //                    KnownNonNull);
  //   }
  // };

  /// Build a CIR function parameter declaration.
  void buildParmDecl(const BlockArgument D, Value Arg, unsigned ArgNo);
};

} // namespace cir
} // namespace mlir
