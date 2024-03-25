#pragma once

#include "mlir/IR/Value.h"
#include "llvm/ADT/STLForwardCompat.h"

namespace mlir {
namespace cir {

/// Contains the address where the return value of a function can be stored, and
/// whether the address is volatile or not.
class ReturnValueSlot {
  Value Addr = {};

  // Return value slot flags
  unsigned IsVolatile : 1;
  unsigned IsUnused : 1;
  unsigned IsExternallyDestructed : 1;

public:
  ReturnValueSlot()
      : IsVolatile(false), IsUnused(false), IsExternallyDestructed(false) {}
  ReturnValueSlot(Value Addr, bool IsVolatile, bool IsUnused = false,
                  bool IsExternallyDestructed = false)
      : Addr(Addr), IsVolatile(IsVolatile), IsUnused(IsUnused),
        IsExternallyDestructed(IsExternallyDestructed) {}

  bool isNull() const { return !Addr; }
  bool isVolatile() const { return IsVolatile; }
  Value getValue() const { return Addr; }
  bool isUnused() const { return IsUnused; }
  bool isExternallyDestructed() const { return IsExternallyDestructed; }
};

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
