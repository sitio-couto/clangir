#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "LoweringFunctionInfo.h"
#include "LoweringModule.h"
#include "LoweringTypes.h"
#include "MissingFeature.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace mlir {
namespace cir {

class X86_64ABIInfo : public ABIInfo {
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

ABIArgInfo X86_64ABIInfo::classifyReturnType(Type RetTy) const {
  llvm_unreachable("NYI");
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

  // If the return value is indirect, then the hidden argument is consuming one
  // integer register.
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
