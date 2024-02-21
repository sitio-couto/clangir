#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "LoweringFunctionInfo.h"
#include "LoweringModule.h"
#include "LoweringTypes.h"
#include "MissingFeature.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace mlir {
namespace cir {

class X86_64ABIInfo : public ABIInfo {
  enum Class {
    Integer = 0,
    SSE,
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    NoClass,
    Memory
  };

  /// Determine the x86_64 register classes in which the given type T should be
  /// passed.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the
  /// containing object.  Some parameters are classified different
  /// depending on whether they straddle an eightbyte boundary.
  ///
  /// \param isNamedArg - Whether the argument in question is a "named"
  /// argument, as used in AMD64-ABI 3.5.7.
  ///
  /// \param IsRegCall - Whether the calling conversion is regcall.
  ///
  /// If a word is unused its result will be NoClass; if a type should
  /// be passed in Memory then at least the classification of \arg Lo
  /// will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will
  /// also be ComplexX87.
  void classify(Type T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool isNamedArg, bool IsRegCall = false) const;

  Type GetINTEGERTypeAtOffset(Type DestTy, unsigned IROffset, Type SourceTy,
                              unsigned SourceOffset) const;

public:
  X86_64ABIInfo(LoweringTypes &CGT, X86AVXABILevel AVXLevel) : ABIInfo(CGT) {}

  void computeInfo(LoweringFunctionInfo &FI) const override;

  ABIArgInfo classifyReturnType(Type RetTy) const;

  ABIArgInfo classifyArgumentType(Type Ty, unsigned freeIntRegs,
                                  unsigned &neededInt, unsigned &neededSSE,
                                  bool isNamedArg, bool IsRegCall) const;
};

class X86_64TargetLoweringInfo : public TargetLoweringInfo {
public:
  X86_64TargetLoweringInfo(LoweringTypes &LM, X86AVXABILevel AVXLevel)
      : TargetLoweringInfo(std::make_unique<X86_64ABIInfo>(LM, AVXLevel)) {
    assert(MissingFeature::Swift());
  }
};

/// The ABI specifies that a value should be passed in an 8-byte GPR.  This
/// means that we either have a scalar or we are talking about the high or low
/// part of an up-to-16-byte struct.  This routine picks the best LLVM IR type
/// to represent this, which may be i64 or may be anything else that the backend
/// will pass in a GPR that works better (e.g. i8, %foo*, etc).
///
/// PrefType is an LLVM IR type that corresponds to (part of) the IR type for
/// the source type.  IROffset is an offset in bytes into the LLVM IR type that
/// the 8-byte value references.  PrefType may be null.
///
/// SourceTy is the source-level type for the entire argument.  SourceOffset is
/// an offset into this that we're processing (which is always either 0 or 8).
///
Type X86_64ABIInfo::GetINTEGERTypeAtOffset(Type DestTy, unsigned IROffset,
                                           Type SourceTy,
                                           unsigned SourceOffset) const {
  llvm_unreachable("NYI");
}

void X86_64ABIInfo::classify(Type Ty, uint64_t OffsetBase, Class &Lo, Class &Hi,
                             bool isNamedArg, bool IsRegCall) const {
  llvm_unreachable("NYI");
}

ABIArgInfo X86_64ABIInfo::classifyReturnType(Type RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.
  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi, true);

  // Check some invariants.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  Type ResType = {};
  switch (Lo) {
  case Integer:
    ResType = GetINTEGERTypeAtOffset(RetTy, 0, RetTy, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && isa<IntType>(ResType)) {
      // Treat an enum type as its underlying type.
      if (!MissingFeature::isEnum())
        llvm_unreachable("NYI");

      if (!MissingFeature::isEnum() && !MissingFeature::promotableForABI())
        llvm_unreachable("NYI");
    }
    break;

  default:
    llvm_unreachable("NYI");
  }

  Type HighPart = {};
  switch (Hi) {

  case NoClass:
    break;

  default:
    llvm_unreachable("NYI");
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    llvm_unreachable("NYI");

  return ABIArgInfo::getDirect(ResType);
}

ABIArgInfo X86_64ABIInfo::classifyArgumentType(Type Ty, unsigned freeIntRegs,
                                               unsigned &neededInt,
                                               unsigned &neededSSE,
                                               bool isNamedArg,
                                               bool IsRegCall = false) const {
  llvm_unreachable("NYI");
}

void X86_64ABIInfo::computeInfo(LoweringFunctionInfo &FI) const {

  const unsigned CallingConv = FI.getCallingConvention();
  // It is possible to force Win64 calling convention on any x86_64 target by
  // using __attribute__((ms_abi)). In such case to correctly emit Win64
  // compatible code delegate this call to WinX86_64ABIInfo::computeInfo.
  if (CallingConv == llvm::CallingConv::Win64) {
    llvm_unreachable("Win64 CC is NYI");
  }

  bool IsRegCall = CallingConv == llvm::CallingConv::X86_RegCall;

  // Keep track of the number of assigned registers.
  unsigned FreeIntRegs = IsRegCall ? 11 : 6;
  unsigned FreeSSERegs = IsRegCall ? 16 : 8;
  unsigned NeededInt = 0, NeededSSE = 0, MaxVectorWidth = 0;

  if (!::mlir::cir::classifyReturnType(getCXXABI(), FI, *this)) {
    if (IsRegCall || !MissingFeature::regCall()) {
      llvm_unreachable("RegCall is NYI");
    } else
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  }

  // If the return value is indirect, then the hidden argument is consuming
  // one integer register.
  if (FI.getReturnInfo().isIndirect())
    llvm_unreachable("NYI");
  else if (NeededSSE && MaxVectorWidth)
    llvm_unreachable("NYI");

  // The chain argument effectively gives us another free register.
  if (!MissingFeature::chainCall())
    llvm_unreachable("NYI");

  unsigned NumRequiredArgs = FI.getNumRequiredArgs();
  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers
  // get assigned (in left-to-right order) for passing as follows...
  unsigned ArgNo = 0;
  for (LoweringFunctionInfo::arg_iterator it = FI.arg_begin(),
                                          ie = FI.arg_end();
       it != ie; ++it, ++ArgNo) {
    bool IsNamedArg = ArgNo < NumRequiredArgs;

    if (IsRegCall && !MissingFeature::regCall())
      llvm_unreachable("NYI");
    else
      it->info = classifyArgumentType(it->type, FreeIntRegs, NeededInt,
                                      NeededSSE, IsNamedArg);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any
    // eightbyte of an argument, the whole argument is passed on the
    // stack. If registers have already been assigned for some
    // eightbytes of such an argument, the assignments get reverted.
    if (FreeIntRegs >= NeededInt && FreeSSERegs >= NeededSSE) {
      FreeIntRegs -= NeededInt;
      FreeSSERegs -= NeededSSE;
      if (!MissingFeature::vector())
        llvm_unreachable("NYI");
    } else {
      llvm_unreachable("Indirect results are NYI");
    }
  }
}

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LoweringModule &CGM, X86AVXABILevel AVXLevel) {
  return std::make_unique<X86_64TargetLoweringInfo>(CGM.getTypes(), AVXLevel);
}

} // namespace cir
} // namespace mlir
