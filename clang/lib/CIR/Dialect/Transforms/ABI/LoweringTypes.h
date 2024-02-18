#pragma once

// Used to replace CodeGenTypes from Clang in ABI lowering.
#include "ABI/LoweringModule.h"

class LoweringModule;

class LoweringTypes {
private:
  LoweringModule &LM;

public:
  LoweringTypes(LoweringModule &LM) : LM(LM){};
  ~LoweringTypes() = default;
};
