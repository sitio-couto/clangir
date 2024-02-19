#pragma once

#include "ABI/MissingFeature.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TrailingObjects.h"
#include <cstddef>
#include <cstdint>

namespace mlir {
namespace cir {

/// Helper class to encapsulate information about how a
/// specific ABI-independent CIR type should be passed to or returned from a
/// function in an ABI-specific way.
class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Pass the argument directly using the normal converted CIR type, or by
    /// coercing to another specified type stored in 'CoerceToType'.  If an
    /// offset is specified (in UIntData), then the argument passed is offset by
    /// some number of bytes in the memory representation. A dummy argument is
    /// emitted before the real argument if the specified type stored in
    /// "PaddingType" is not zero.
    Direct,

    /// Valid only for integer argument types. Same as 'direct' but also emit a
    /// zero/sign extension attribute.
    Extend,
  };

private:
  Kind kind;
  mlir::Type typeData;

public:
  ABIArgInfo(Kind kind = Direct) : kind(kind){};
  ~ABIArgInfo() = default;

  Type getCoerceToType() const {
    assert(canHaveCoerceToType() && "Invalid kind!");
    return typeData;
  }

  Kind getKind() const { return kind; }
  bool isDirect() const { return kind == Direct; }
  bool isExtend() const { return kind == Extend; }
  bool isCoerceAndExpand() const {
    MissingFeature::isCoerceAndExpand();
    llvm_unreachable("NYI");
  }

  bool canHaveCoerceToType() const {
    return isDirect() || isExtend() || isCoerceAndExpand();
  }

  bool getCanBeFlattened() const {
    assert(MissingFeature::canBeFlattened());
    return false;
  }
};

// Implementation detail of CGFunctionInfo, factored out so it can be named
// in the TrailingObjects base class of CGFunctionInfo.
struct LoweringFunctionInfoArgInfo {
  mlir::Type type;
  ABIArgInfo info;
};

class LoweringFunctionInfo final
    : private llvm::TrailingObjects<LoweringFunctionInfo,
                                    LoweringFunctionInfoArgInfo> {
private:
  typedef LoweringFunctionInfoArgInfo ArgInfo;

  const ArgInfo *getArgsBuffer() const { return getTrailingObjects<ArgInfo>(); }

  unsigned NumArgs;

  LoweringFunctionInfo() = default;

public:
  static LoweringFunctionInfo *create(ArrayRef<mlir::Type> argTypes) {
    // TODO(cir): Add assertions?
    assert(MissingFeature::extParamInfo());
    void *buffer = operator new(totalSizeToAlloc<ArgInfo>(argTypes.size() + 1));

    LoweringFunctionInfo *FI = new (buffer) LoweringFunctionInfo();
    FI->NumArgs = argTypes.size();

    return FI;
  };
  void operator delete(void *p) { ::operator delete(p); }

  // Friending class TrailingObjects is apparently not good enough for MSVC,
  // so these have to be public.
  friend class TrailingObjects;
  size_t numTrailingObjects(OverloadToken<ArgInfo>) const {
    return NumArgs + 1;
  }

  typedef const ArgInfo *const_arg_iterator;

  const_arg_iterator arg_begin() const { return getArgsBuffer() + 1; }

  bool isVariadic() const {
    assert(MissingFeature::variadicFunctions());
    return false;
  }
  unsigned getNumRequiredArgs() const { llvm_unreachable("NYI"); }

  const ABIArgInfo &getReturnInfo() const { return {}; }
};

} // namespace cir
} // namespace mlir
