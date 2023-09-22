//===- CIRTypes.h - MLIR CIR Types ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_IR_CIRTYPES_H_
#define MLIR_DIALECT_CIR_IR_CIRTYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include <optional>

//===----------------------------------------------------------------------===//
// CIR Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

namespace mlir {
namespace cir {

// Custom type storages.
namespace detail {
struct StructTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// StructType
//
// The base type for all RecordDecls.
//
//===----------------------------------------------------------------------===//

class StructType
    : public Type::TypeBase<StructType, Type, detail::StructTypeStorage,
                            DataLayoutTypeInterface::Trait> {
public:
  using Base::Base;
  enum RecordKind : uint32_t { Class, Union, Struct };

  /// Get an identfied and complete struct type.
  static StructType get(MLIRContext *context, ArrayRef<Type> members,
                        mlir::StringAttr name, bool packed,
                        StructType::RecordKind kind,
                        ASTRecordDeclInterface ast);
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, ArrayRef<Type> members,
                               mlir::StringAttr name, bool packed,
                               StructType::RecordKind kind,
                               ASTRecordDeclInterface ast);

  /// Get an identfied and incomplete struct type.
  static StructType get(MLIRContext *context, mlir::StringAttr name,
                        bool packed, StructType::RecordKind kind,
                        ASTRecordDeclInterface ast);
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, mlir::StringAttr name,
                               bool packed, StructType::RecordKind kind,
                               ASTRecordDeclInterface ast);

  // Get a anonymous struct type (always complete).
  static StructType get(MLIRContext *context, ArrayRef<Type> members,
                        bool packed, StructType::RecordKind kind,
                        ASTRecordDeclInterface ast);
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, ArrayRef<Type> members,
                               bool packed, StructType::RecordKind kind,
                               ASTRecordDeclInterface ast);

  //
  // Class methods.
  //

  static constexpr StringLiteral getMnemonic() { return {"struct"}; }
  static Type parse(AsmParser &odsParser);
  static void print(cir::StructType type, AsmPrinter &odsPrinter);

  //
  // Accessors.
  //

  bool getPacked() const;
  StructType::RecordKind getKind() const;
  ASTRecordDeclInterface getAst() const;
  ArrayRef<Type> getMembers() const;
  StringAttr getName() const;
  void dropAst();

  //
  // Pseudo accessors.
  //

  /// Return the number of members in the struct.
  size_t getNumElements() const { return getMembers().size(); }
  /// Return the member with the largest bit-length.
  mlir::Type getLargestMember(const DataLayout &dataLayout) const;
  /// Return the name of the struct prefixed with its kind.
  std::string getPrefixedName() {
    const auto name = getName().getValue().str();
    switch (getKind()) {
    case RecordKind::Class:
      return "class." + name;
    case RecordKind::Union:
      return "union." + name;
    case RecordKind::Struct:
      return "struct." + name;
    }
  }

  //
  // Predicates.
  //

  /// Return whether this is forward declaration.
  bool isIncomplete() const;
  /// Return whether this is a full declaration.
  bool isComplete() const { return !isIncomplete(); };
  /// Return whether this struct is padded.
  bool isPadded(const DataLayout &dataLayout) const;
  /// Return whether this is a class declaration.
  bool isClass() const { return getKind() == RecordKind::Class; }
  /// Return whether this is a union declaration.
  bool isUnion() const { return getKind() == RecordKind::Union; }
  /// Return whether this is a struct declaration.
  bool isStruct() const { return getKind() == RecordKind::Struct; }

private:
  // Attributes used to store struct layout information. These are lazily
  // computed when queried, and once computed they will be stored for future
  // queries.
  mutable std::optional<unsigned> size{}, align{};
  mutable std::optional<bool> padded{};
  mutable mlir::Type largestMember{};

  /// Compute the size and alignment of this struct. If it was already computed,
  /// return the cached values.
  void computeSizeAndAlignment(const DataLayout &dataLayout) const;

public:
  //
  // DataLayoutTypeInterface::Trait methods.
  //

  unsigned getTypeSizeInBits(const DataLayout &dataLayout,
                             DataLayoutEntryListRef params) const;
  unsigned getABIAlignment(const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const;
  unsigned getPreferredAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const;
};

} // namespace cir
} // namespace mlir

#endif // MLIR_DIALECT_CIR_IR_CIRTYPES_H_
