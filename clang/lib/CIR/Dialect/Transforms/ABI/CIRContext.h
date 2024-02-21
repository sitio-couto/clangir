#pragma once

#include "mlir/IR/Types.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace mlir {
namespace cir {

enum class AlignRequirementKind {
  /// The alignment was not explicit in code.
  None,

  /// The alignment comes from an alignment attribute on a typedef.
  RequiredByTypedef,

  /// The alignment comes from an alignment attribute on a record type.
  RequiredByRecord,

  /// The alignment comes from an alignment attribute on a enum type.
  RequiredByEnum,
};

struct TypeInfo {
  uint64_t Width = 0;
  unsigned Align = 0;
  AlignRequirementKind AlignRequirement;

  TypeInfo() : AlignRequirement(AlignRequirementKind::None) {}
  TypeInfo(uint64_t Width, unsigned Align,
           AlignRequirementKind AlignRequirement)
      : Width(Width), Align(Align), AlignRequirement(AlignRequirement) {}
  bool isAlignRequired() {
    return AlignRequirement != AlignRequirementKind::None;
  }
};

class CIRContext : public llvm::RefCountedBase<CIRContext> {

private:
  TypeInfo getTypeInfoImpl(const Type T) const;

  const clang::TargetInfo *Target = nullptr;

public:
  //===--------------------------------------------------------------------===//
  //                         Type Sizing and Analysis
  //===--------------------------------------------------------------------===//

  /// Get the size and alignment of the specified complete type in bits.
  TypeInfo getTypeInfo(Type T) const;

  /// Return the size of the specified (complete) type \p T, in bits.
  uint64_t getTypeSize(Type T) const { return getTypeInfo(T).Width; }
};

} // namespace cir
} // namespace mlir