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
    : public ::mlir::Type::TypeBase<StructType, ::mlir::Type,
                                    detail::StructTypeStorage,
                                    ::mlir::DataLayoutTypeInterface::Trait> {
public:
  using Base::Base;
  enum RecordKind : uint32_t { Class, Union, Struct };

private:
  // All these support lazily computation and storage
  // for the struct size and alignment.
  mutable std::optional<unsigned> size{}, align{};
  mutable std::optional<bool> padded{};
  mutable mlir::Type largestMember{};
  void computeSizeAndAlignment(const ::mlir::DataLayout &dataLayout) const;

public:
  void dropAst();
  size_t getNumElements() const { return getMembers().size(); }
  bool isOpaque() const { return !getBody(); }
  bool isPadded(const ::mlir::DataLayout &dataLayout) const;

  std::string getPrefixedName() {
    const auto name = getTypeName().getValue().str();
    switch (getKind()) {
    case RecordKind::Class:
      return "class." + name;
    case RecordKind::Union:
      return "union." + name;
    case RecordKind::Struct:
      return "struct." + name;
    }
  }

  /// Return the member with the largest bit-length.
  mlir::Type getLargestMember(const ::mlir::DataLayout &dataLayout) const;

  /// Return whether this is a class declaration.
  bool isClass() const { return getKind() == RecordKind::Class; }

  /// Return whether this is a union declaration.
  bool isUnion() const { return getKind() == RecordKind::Union; }

  /// Return whether this is a struct declaration.
  bool isStruct() const { return getKind() == RecordKind::Struct; }
  static StructType get(::mlir::MLIRContext *context,
                        ::llvm::ArrayRef<mlir::Type> members,
                        mlir::StringAttr typeName, bool body, bool packed,
                        mlir::cir::StructType::RecordKind kind,
                        std::optional<ASTRecordDeclInterface> ast);
  static constexpr ::llvm::StringLiteral getMnemonic() { return {"struct"}; }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  static void print(::mlir::cir::StructType type,
                    ::mlir::AsmPrinter &odsPrinter);
  ::llvm::ArrayRef<mlir::Type> getMembers() const;
  mlir::StringAttr getTypeName() const;
  bool getBody() const;
  bool getPacked() const;
  mlir::cir::StructType::RecordKind getKind() const;
  std::optional<ASTRecordDeclInterface> getAst() const;
  unsigned getTypeSizeInBits(const ::mlir::DataLayout &dataLayout,
                             ::mlir::DataLayoutEntryListRef params) const;
  unsigned getABIAlignment(const ::mlir::DataLayout &dataLayout,
                           ::mlir::DataLayoutEntryListRef params) const;
  unsigned getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const;
};

} // namespace cir
} // namespace mlir

#endif // MLIR_DIALECT_CIR_IR_CIRTYPES_H_
