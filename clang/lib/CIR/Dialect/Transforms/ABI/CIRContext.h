#pragma once

#include "CIRRecordLayout.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "clang/AST/Type.h"
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
  mutable SmallVector<Type, 0> Types;

  TypeInfo getTypeInfoImpl(const Type T) const;

  const clang::TargetInfo *Target = nullptr;
  const clang::TargetInfo *AuxTarget = nullptr;

  /// MLIR context to be used when creating types.
  MLIRContext *MLIRCtx;

  /// The language options used to create the AST associated with
  /// this ASTContext object.
  clang::LangOptions &LangOpts;

  //===--------------------------------------------------------------------===//
  //                         Built-in Types
  //===--------------------------------------------------------------------===//

  Type CharTy;

public:
  CIRContext(MLIRContext *MLIRCtx, clang::LangOptions &LOpts);
  CIRContext(const CIRContext &) = delete;
  CIRContext &operator=(const CIRContext &) = delete;
  ~CIRContext();

  /// Initialize built-in types.
  ///
  /// This routine may only be invoked once for a given ASTContext object.
  /// It is normally invoked after ASTContext construction.
  ///
  /// \param Target The target
  void initBuiltinTypes(const clang::TargetInfo &Target,
                        const clang::TargetInfo *AuxTarget = nullptr);

private:
  void initBuiltinType(Type &R, clang::BuiltinType::Kind K);

public:
  const clang::TargetInfo &getTargetInfo() const { return *Target; }

  const clang::LangOptions &getLangOpts() const { return LangOpts; }

  MLIRContext *getMLIRContext() const { return MLIRCtx; }

  //===--------------------------------------------------------------------===//
  //                         Type Sizing and Analysis
  //===--------------------------------------------------------------------===//

  /// Get the size and alignment of the specified complete type in bits.
  TypeInfo getTypeInfo(Type T) const;

  /// Return the size of the specified (complete) type \p T, in bits.
  uint64_t getTypeSize(Type T) const { return getTypeInfo(T).Width; }

  /// Return the size of the character type, in bits.
  uint64_t getCharWidth() const { return getTypeSize(CharTy); }

  /// Convert a size in bits to a size in characters.
  clang::CharUnits toCharUnitsFromBits(int64_t BitSize) const;

  /// Convert a size in characters to a size in bits.
  int64_t toBits(clang::CharUnits CharSize) const;

  /// Get or compute information about the layout of the specified
  /// record (struct/union/class) \p D, which indicates its size and field
  /// position information.
  const CIRRecordLayout &getCIRRecordLayout(const Type D) const;
};

} // namespace cir
} // namespace mlir