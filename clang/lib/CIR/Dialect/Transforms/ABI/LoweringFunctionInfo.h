#pragma once

#include "MissingFeature.h"
#include "mlir/IR/Types.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
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

    /// Not yet supported.
    Indirect,
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

/// A class for recording the number of arguments that a function
/// signature requires.
class RequiredArgs {
  /// The number of required arguments, or ~0 if the signature does
  /// not permit optional arguments.
  unsigned NumRequired;

public:
  enum All_t { All };

  RequiredArgs(All_t _) : NumRequired(~0U) {}
  explicit RequiredArgs(unsigned n) : NumRequired(n) { assert(n != ~0U); }

  /// Compute the arguments required by the given formal prototype,
  /// given that there may be some additional, non-formal arguments
  /// in play.
  ///
  /// If FD is not null, this will consider pass_object_size params in FD.
  static RequiredArgs forPrototypePlus(const FuncType prototype,
                                       unsigned additional) {
    if (!prototype.isVarArg())
      return All;

    llvm_unreachable("Variadic function is NYI");
  }

  static RequiredArgs forPrototype(const FuncType prototype) {
    return forPrototypePlus(prototype, 0);
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
  typedef LoweringFunctionInfoArgInfo ArgInfo;

  /// The LLVM::CallingConv to use for this function (as specified by the
  /// user).
  unsigned CallingConvention : 8;

  /// The LLVM::CallingConv to actually use for this function, which may
  /// depend on the ABI.
  unsigned EffectiveCallingConvention : 8;

  /// Whether this is an instance method.
  unsigned InstanceMethod : 1;

  /// Whether this is a chain call.
  unsigned ChainCall : 1;

  /// Whether this function is called by forwarding arguments.
  /// This doesn't support inalloca or varargs.
  unsigned DelegateCall : 1;

  RequiredArgs Required;

  unsigned NumArgs;

  const ArgInfo *getArgsBuffer() const { return getTrailingObjects<ArgInfo>(); }
  ArgInfo *getArgsBuffer() { return getTrailingObjects<ArgInfo>(); }

  LoweringFunctionInfo() : Required(RequiredArgs::All) {}

public:
  static LoweringFunctionInfo *create(unsigned llvmCC, bool instanceMethod,
                                      bool chainCall, bool delegateCall,
                                      Type resultType,
                                      ArrayRef<mlir::Type> argTypes,
                                      RequiredArgs required) {
    // TODO(cir): Add assertions?
    assert(MissingFeature::extParamInfo());
    void *buffer = operator new(totalSizeToAlloc<ArgInfo>(argTypes.size() + 1));

    LoweringFunctionInfo *FI = new (buffer) LoweringFunctionInfo();
    FI->CallingConvention = llvmCC;
    FI->EffectiveCallingConvention = llvmCC;
    FI->InstanceMethod = instanceMethod;
    FI->ChainCall = chainCall;
    FI->DelegateCall = delegateCall;
    FI->Required = required;
    FI->NumArgs = argTypes.size();
    FI->getArgsBuffer()[0].type = resultType;
    for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
      FI->getArgsBuffer()[i + 1].type = argTypes[i];

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
  const_arg_iterator arg_end() const { return getArgsBuffer() + 1 + NumArgs; }

  unsigned arg_size() const { return NumArgs; }

  bool isVariadic() const {
    assert(MissingFeature::variadicFunctions());
    return false;
  }
  unsigned getNumRequiredArgs() const {
    if (isVariadic())
      llvm_unreachable("NYI");
    return arg_size();
  }

  const ABIArgInfo &getReturnInfo() const { return ABIArgInfo{}; }
};

} // namespace cir
} // namespace mlir
