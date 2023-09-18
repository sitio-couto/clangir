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
#include "llvm/ADT/Hashing.h"

namespace mlir {
namespace cir {
namespace detail {

struct StructTypeStorage : public TypeStorage {

  /// Helper class for managing StructType key-related operations. It stores
  /// information only temporarily.
  struct KeyTy {
    friend StructTypeStorage;

  private:
    ArrayRef<Type> members;
    StringAttr typeName;
    bool body;
    bool packed;
    StructType::RecordKind kind;
    std::optional<ASTRecordDeclInterface> ast;

  public:
    KeyTy(ArrayRef<Type> members, StringAttr typeName, bool body, bool packed,
          StructType::RecordKind kind,
          std::optional<ASTRecordDeclInterface> ast)
        : members(members), typeName(typeName), body(body), packed(packed),
          kind(kind), ast(ast) {}

    bool operator==(const KeyTy &other) const {
      return (members == other.members) && (typeName == other.typeName) &&
             (body == other.body) && (packed == other.packed) &&
             (kind == other.kind) && (ast == other.ast);
    }

    llvm::hash_code hashValue() const {
      return hash_combine(members, typeName, body, packed, kind, ast);
    }

    /// Copies dynamically-sized components of the key into the given allocator.
    KeyTy copyIntoAllocator(TypeStorageAllocator &allocator) const {
      return KeyTy(allocator.copyInto(members), typeName, body, packed, kind,
                   ast);
    }
  };

  // Storage for the types parameters and attributes.
  ArrayRef<Type> members;
  StringAttr typeName;
  bool body;
  bool packed;
  StructType::RecordKind kind;
  std::optional<ASTRecordDeclInterface> ast;

  /// Constructs the storage from the given key.
  StructTypeStorage(const KeyTy &key)
      : members(key.members), typeName(key.typeName), body(key.body),
        packed(key.packed), kind(key.kind), ast(key.ast) {}

  /// Get the storage instance as a key.
  KeyTy getAsKey() const {
    return KeyTy(members, typeName, body, packed, kind, ast);
  }

  //
  // Hooks for the uniquing infrastructure.
  //

  bool operator==(const KeyTy &key) const { return getAsKey() == key; }
  static llvm::hash_code hashKey(const KeyTy &key) { return key.hashValue(); }
  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    StructTypeStorage *location = allocator.allocate<StructTypeStorage>();
    return new (location) StructTypeStorage(key.copyIntoAllocator(allocator));
  }
};

} // namespace detail
} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_DIALECT_IR_SIMPLETYPESDETAILS_H
