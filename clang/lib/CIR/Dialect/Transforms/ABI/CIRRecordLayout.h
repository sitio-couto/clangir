#pragma once

#include "clang/AST/CharUnits.h"
#include <cstdint>
#include <vector>

namespace mlir {
namespace cir {

/// This class contains layout information for one RecordDecl, which is a
/// struct/union/class.  The decl represented must be a definition, not a
/// forward declaration. This class is also used to contain layout information
/// for one ObjCInterfaceDecl.
/// FIXME - Find appropriate name. These objects are managed by CIRContext.
class CIRRecordLayout {

private:
  friend class CIRContext;

  /// Size of record in characters.
  clang::CharUnits Size;

  // Alignment of record in characters.
  clang::CharUnits Alignment;

  /// Array of field offsets in bits.
  /// FIXME(cir): Create a custom CIRVector instead?
  std::vector<uint64_t> FieldOffsets;

  // Constructor for C++ records.
  CIRRecordLayout();

  ~CIRRecordLayout() = default;

public:

  /// Get the record alignment in characters.
  clang::CharUnits getAlignment() const { return Alignment; }

  /// Get the record size in characters.
  clang::CharUnits getSize() const { return Size; }

  /// Get the offset of the given field index, in bits.
  uint64_t getFieldOffset(unsigned FieldNo) const {
    return FieldOffsets[FieldNo];
  }
};

} // namespace cir
} // namespace mlir
