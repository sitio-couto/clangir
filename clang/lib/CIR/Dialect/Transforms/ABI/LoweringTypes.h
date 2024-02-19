#pragma once

// Used to replace CodeGenTypes from Clang in ABI lowering.
#include "ABI/LoweringFunctionInfo.h"
#include "ABI/CIRToCIRArgMapping.h"
#include "ABI/MissingFeature.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

class LoweringModule;

class LoweringTypes {
private:
  LoweringModule &LM;

  // FIXME(cir): We should be able to query this from LM.
  MLIRContext *ctx;
  MLIRContext *getContext() { return ctx; }

public:
  LoweringTypes(LoweringModule &LM, MLIRContext *ctx) : LM(LM), ctx(ctx){};
  ~LoweringTypes() = default;

  const LoweringFunctionInfo &arrangeGlobalDeclaration(FuncOp GD) { return {}; }

  FuncType getFunctionType(const LoweringFunctionInfo &FI) {

    assert(MissingFeature::recursiveFunctionProcessing());

    mlir::Type resultType = {};
    const ABIArgInfo &retAI = FI.getReturnInfo();
    switch (retAI.getKind()) {
    case ABIArgInfo::Direct:
      resultType = retAI.getCoerceToType();
      break;
    default:
      llvm_unreachable("Missing ABIArgInfo::Kind");
    }

    CIRToCIRArgMapping IRFunctionArgs(getContext(), FI, true);
    SmallVector<Type, 8> ArgTypes;

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

    return FuncType::get(getContext(), ArgTypes, resultType, FI.isVariadic());
  }
};

} // namespace cir
} // namespace mlir
