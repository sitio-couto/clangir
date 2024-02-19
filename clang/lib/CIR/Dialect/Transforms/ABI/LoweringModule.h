#pragma once

#include "ABI/LoweringTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace cir {

/// Replaces CodeGenModule from Clang in ABI lowering.
class LoweringModule {
private:
  ModuleOp module;
  LoweringTypes types;

public:
  LoweringModule(ModuleOp &module)
      : module(module), types(*this, module.getContext()){};
  ~LoweringModule() = default;

  LoweringTypes &getTypes() { return types; }
  MLIRContext *getContext() { return module.getContext(); }
};

} // namespace cir
} // namespace mlir
