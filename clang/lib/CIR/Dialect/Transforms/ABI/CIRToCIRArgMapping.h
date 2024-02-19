#pragma once

#include "ABI/LoweringFunctionInfo.h"
#include "ABI/MissingFeature.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
namespace mlir {
namespace cir {

/// Encapsulates information about the way function arguments from
/// LoweringFunctionInfo should be passed to actual CIR function.
class CIRToCIRArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned TotalIRArgs;

  /// Arguments of CIR function corresponding to single CIR argument.
  struct IRArgs {
    unsigned PaddingArgIndex;
    // Argument is expanded to IR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex;
    unsigned NumberOfArgs;

    IRArgs()
        : PaddingArgIndex(InvalidIndex), FirstArgIndex(InvalidIndex),
          NumberOfArgs(0) {}
  };

  llvm::SmallVector<IRArgs, 8> ArgInfo;

public:
  CIRToCIRArgMapping(const MLIRContext *ctx, const LoweringFunctionInfo &FI,
                     bool onlyRequiredArgs = false)
      : ArgInfo(onlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size()) {
    construct(ctx, FI, onlyRequiredArgs);
  };

  void construct(const MLIRContext *ctx, const LoweringFunctionInfo &FI,
                 bool onlyRequiredArgs = false) {
    unsigned IRArgNo = 0;
    bool SwapThisWithSRet = false;
    const ABIArgInfo &RetAI = FI.getReturnInfo();

    if (RetAI.getKind() == ABIArgInfo::Indirect) {
      llvm_unreachable("NYI");
    }

    unsigned ArgNo = 0;
    unsigned NumArgs =
        onlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size();
    for (LoweringFunctionInfo::const_arg_iterator I = FI.arg_begin();
         ArgNo < NumArgs; ++I, ++ArgNo) {
      assert(I != FI.arg_end());
      Type ArgType = I->type;
      const ABIArgInfo &AI = I->info;
      // Collect data about IR arguments corresponding to Clang argument ArgNo.
      auto &IRArgs = ArgInfo[ArgNo];

      if (!MissingFeature::argumentPadding()) {
        llvm_unreachable("NYI");
      }

      switch (AI.getKind()) {
      case ABIArgInfo::Extend:
      case ABIArgInfo::Direct: {
        // FIXME(cir): handle sseregparm someday...
        StructType STy = dyn_cast<StructType>(AI.getCoerceToType());
        if (AI.isDirect() && AI.getCanBeFlattened() && STy) {
          IRArgs.NumberOfArgs = STy.getNumElements();
        } else {
          IRArgs.NumberOfArgs = 1;
        }
        break;
      }
      default:
        llvm_unreachable("Missing ABIArgInfo::Kind");
      }

      if (IRArgs.NumberOfArgs > 0) {
        IRArgs.FirstArgIndex = IRArgNo;
        IRArgNo += IRArgs.NumberOfArgs;
      }

      // Skip over the sret parameter when it comes second.  We already handled
      // it above.
      if (IRArgNo == 1 && SwapThisWithSRet)
        IRArgNo++;
    }
    assert(ArgNo == ArgInfo.size());

    if (!MissingFeature::inallocaArgument()) {
      llvm_unreachable("NYI");
    }

    TotalIRArgs = IRArgNo;
  }

  /// Returns index of first IR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getIRArgs(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return std::make_pair(ArgInfo[ArgNo].FirstArgIndex,
                          ArgInfo[ArgNo].NumberOfArgs);
  }
};

} // namespace cir
} // namespace mlir
