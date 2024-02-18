#pragma once

// Used to replace CodeGenModule from Clang.
#include "ABI/LoweringTypes.h"

class LoweringTypes;

class LoweringModule {
private:
  LoweringTypes types;

public:
  LoweringModule() : types(*this) {};
  ~LoweringModule() = default;
};
