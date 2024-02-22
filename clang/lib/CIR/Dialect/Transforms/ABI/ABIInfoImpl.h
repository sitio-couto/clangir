#pragma once

#include "ABIInfo.h"
#include "CIRCXXABI.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LoweringFunctionInfo &FI,
                        const ABIInfo &Info);

Type useFirstFieldIfTransparentUnion(Type Ty);

CIRCXXABI::RecordArgABI getRecordArgABI(const StructType RT, CIRCXXABI &CXXABI);

} // namespace cir
} // namespace mlir
