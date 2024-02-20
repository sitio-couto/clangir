#pragma once

#include "LoweringModule.h"
#include "LoweringTypes.h"
#include "TargetLoweringInfo.h"
#include <memory>

namespace mlir {
namespace cir {

/// The AVX ABI level for X86 targets.
enum class X86AVXABILevel {
  None,
  AVX,
  AVX512,
};

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LoweringModule &CGM, X86AVXABILevel AVXLevel);

} // namespace cir
} // namespace mlir
