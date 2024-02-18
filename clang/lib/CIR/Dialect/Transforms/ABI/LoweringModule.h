#pragma once

// Used to replace CodeGenModule from Clang.
#include "ABI/LoweringTypes.h"

namespace mlir {
namespace cir {

class LoweringTypes;

class LoweringModule {
private:
  LoweringTypes types;

public:
  LoweringModule() : types(*this){};
  ~LoweringModule() = default;

  LoweringTypes &getTypes() { return types; }
};

} // namespace cir
} // namespace mlir
