//===- CIRTypesDetails.h - MLIR CIR Types Details ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares implementation details regarding CIR types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIALECT_IR_SIMPLETYPESDETAILS_H
#define LLVM_CLANG_DIALECT_IR_SIMPLETYPESDETAILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeSupport.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace mlir {
namespace cir {
namespace detail {

struct StructTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<mlir::Type>, mlir::StringAttr, bool,
                           bool, mlir::cir::StructType::RecordKind,
                           std::optional<ASTRecordDeclInterface>>;
  StructTypeStorage(::llvm::ArrayRef<mlir::Type> members,
                    mlir::StringAttr typeName, bool body, bool packed,
                    mlir::cir::StructType::RecordKind kind,
                    std::optional<ASTRecordDeclInterface> ast)
      : members(members), typeName(typeName), body(body), packed(packed),
        kind(kind), ast(ast) {}

  KeyTy getAsKey() const {
    return KeyTy(members, typeName, body, packed, kind, ast);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (members == std::get<0>(tblgenKey)) &&
           (typeName == std::get<1>(tblgenKey)) &&
           (body == std::get<2>(tblgenKey)) &&
           (packed == std::get<3>(tblgenKey)) &&
           (kind == std::get<4>(tblgenKey)) && (ast == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey),
                                std::get<2>(tblgenKey), std::get<3>(tblgenKey),
                                std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static StructTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &tblgenKey) {
    auto members = std::get<0>(tblgenKey);
    auto typeName = std::get<1>(tblgenKey);
    auto body = std::get<2>(tblgenKey);
    auto packed = std::get<3>(tblgenKey);
    auto kind = std::get<4>(tblgenKey);
    auto ast = std::get<5>(tblgenKey);
    members = allocator.copyInto(members);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(members, typeName, body, packed, kind, ast);
  }

  ::llvm::ArrayRef<mlir::Type> members;
  mlir::StringAttr typeName;
  bool body;
  bool packed;
  mlir::cir::StructType::RecordKind kind;
  std::optional<mlir::cir::ASTRecordDeclInterface> ast;
};

} // namespace detail
} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_DIALECT_IR_SIMPLETYPESDETAILS_H
