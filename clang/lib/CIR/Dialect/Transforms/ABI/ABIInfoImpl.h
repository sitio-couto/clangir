#pragma once

#include "ABIInfo.h"
#include "CIRCXXABI.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LoweringFunctionInfo &FI,
                        const ABIInfo &Info);

} // namespace cir
} // namespace mlir
