#pragma once

#include "llvm/ADT/STLForwardCompat.h"

namespace mlir {
namespace cir {

enum class FnInfoOpts {
  None = 0,
  IsInstanceMethod = 1 << 0,
  IsChainCall = 1 << 1,
  IsDelegateCall = 1 << 2,
};

inline FnInfoOpts operator|(FnInfoOpts A, FnInfoOpts B) {
  return static_cast<FnInfoOpts>(llvm::to_underlying(A) |
                                 llvm::to_underlying(B));
}

inline FnInfoOpts operator&(FnInfoOpts A, FnInfoOpts B) {
  return static_cast<FnInfoOpts>(llvm::to_underlying(A) &
                                 llvm::to_underlying(B));
}

inline FnInfoOpts operator|=(FnInfoOpts A, FnInfoOpts B) {
  A = A | B;
  return A;
}

inline FnInfoOpts operator&=(FnInfoOpts A, FnInfoOpts B) {
  A = A & B;
  return A;
}

} // namespace cir
} // namespace mlir
