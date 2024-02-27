#pragma once

#include "MissingFeature.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

using Align = llvm::Align;
using Error = llvm::Error;

namespace mlir {
namespace cir {

class StructLayout;

/// Enum used to categorize the alignment types stored by LayoutAlignElem
enum AlignTypeEnum {
  INTEGER_ALIGN = 'i',
  VECTOR_ALIGN = 'v',
  FLOAT_ALIGN = 'f',
  AGGREGATE_ALIGN = 'a'
};

// FIXME: Currently the DataLayout string carries a "preferred alignment"
// for types. As the DataLayout is module/global, this should likely be
// sunk down to an FTTI element that is queried rather than a global
// preference.

/// Layout alignment element.
///
/// Stores the alignment data associated with a given type bit width.
///
/// \note The unusual order of elements in the structure attempts to reduce
/// padding and make the structure slightly more cache friendly.
struct LayoutAlignElem {
  uint32_t TypeBitWidth;
  llvm::Align ABIAlign;
  llvm::Align PrefAlign;

  static LayoutAlignElem get(llvm::Align ABIAlign, llvm::Align PrefAlign,
                             uint32_t BitWidth);

  bool operator==(const LayoutAlignElem &rhs) const;
};

/// Layout pointer alignment element.
///
/// Stores the alignment data associated with a given pointer and address space.
///
/// \note The unusual order of elements in the structure attempts to reduce
/// padding and make the structure slightly more cache friendly.
struct PointerAlignElem {
  Align ABIAlign;
  Align PrefAlign;
  uint32_t TypeBitWidth;
  uint32_t AddressSpace;
  uint32_t IndexBitWidth;

  /// Initializer
  static PointerAlignElem getInBits(uint32_t AddressSpace, Align ABIAlign,
                                    Align PrefAlign, uint32_t TypeBitWidth,
                                    uint32_t IndexBitWidth);

  bool operator==(const PointerAlignElem &rhs) const;
};

/// A parsed version of the target data layout string in and methods for
/// querying it.
///
/// The target data layout string is specified *by the target* - a frontend
/// generating LLVM IR is required to generate the right target data for the
/// target being codegen'd to.
class CIRDataLayout {
public:
  enum class FunctionPtrAlignType {
    /// The function pointer alignment is independent of the function alignment.
    Independent,
    /// The function pointer alignment is a multiple of the function alignment.
    MultipleOfFunctionAlign,
  };

private:
  /// Defaults to false.
  bool BigEndian;

  unsigned AllocaAddrSpace;
  llvm::MaybeAlign StackNaturalAlign;
  unsigned ProgramAddrSpace;
  unsigned DefaultGlobalsAddrSpace;

  llvm::MaybeAlign FunctionPtrAlign;
  FunctionPtrAlignType TheFunctionPtrAlignType;

  enum ManglingModeT {
    MM_None,
    MM_ELF,
    MM_MachO,
    MM_WinCOFF,
    MM_WinCOFFX86,
    MM_GOFF,
    MM_Mips,
    MM_XCOFF
  };
  ManglingModeT ManglingMode;

  SmallVector<unsigned char, 8> LegalIntWidths;

  /// Primitive type alignment data. This is sorted by type and bit
  /// width during construction.
  using AlignmentsTy = SmallVector<LayoutAlignElem, 4>;
  AlignmentsTy IntAlignments;
  AlignmentsTy FloatAlignments;
  AlignmentsTy VectorAlignments;
  LayoutAlignElem StructAlignment;

  /// The string representation used to create this DataLayout
  std::string StringRepresentation;

  using PointersTy = SmallVector<PointerAlignElem, 8>;
  PointersTy Pointers;

  // The StructType -> StructLayout map.
  mutable void *LayoutMap = nullptr;

  /// Attempts to set the alignment of the given type. Returns an error
  /// description on failure.
  Error setAlignment(AlignTypeEnum AlignType, Align ABIAlign, Align PrefAlign,
                     uint32_t BitWidth);

public:
  /// Constructs a DataLayout from a specification string. See reset().
  explicit CIRDataLayout(StringRef dataLayout) { reset(dataLayout); }

  /// Parse a data layout string (with fallback to default values).
  void reset(StringRef dataLayout);

  /// Attempts to set the alignment of a pointer in the given address space.
  /// Returns an error description on failure.
  Error setPointerAlignmentInBits(uint32_t AddrSpace, Align ABIAlign,
                                  Align PrefAlign, uint32_t TypeBitWidth,
                                  uint32_t IndexBitWidth);

  /// Attempts to parse a target data specification string and reports an error
  /// if the string is malformed.
  Error parseSpecifier(StringRef Desc);

  // Free all internal data structures.
  void clear();

  /// Returns a StructLayout object, indicating the alignment of the
  /// struct, its size, and the offsets of its fields.
  ///
  /// Note that this information is lazily cached.
  const StructLayout *getStructLayout(StructType Ty) const;

  /// Internal helper to get alignment for integer of given bitwidth.
  Align getIntegerAlignment(uint32_t BitWidth, bool abi_or_pref) const;

  /// Internal helper method that returns requested alignment for type.
  Align getAlignment(Type Ty, bool abi_or_pref) const;

  Align getABITypeAlign(Type Ty) const { return getAlignment(Ty, true); }

  /// Returns the maximum number of bytes that may be overwritten by
  /// storing the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 5 for i36 and 10 for x86_fp80.
  llvm::TypeSize getTypeStoreSize(Type Ty) const {
    llvm::TypeSize BaseSize = getTypeSizeInBits(Ty);
    return {llvm::divideCeil(BaseSize.getKnownMinValue(), 8),
            BaseSize.isScalable()};
  }

  /// Returns the offset in bytes between successive objects of the
  /// specified type, including alignment padding.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// This is the amount that alloca reserves for this type. For example,
  /// returns 12 or 16 for x86_fp80, depending on alignment.
  llvm::TypeSize getTypeAllocSize(Type Ty) const {
    // Round up to the next alignment boundary.
    return alignTo(getTypeStoreSize(Ty), getABITypeAlign(Ty).value());
  }

  // The implementation of this method is provided inline as it is particularly
  // well suited to constant folding when called on a specific Type subclass.
  llvm::TypeSize getTypeSizeInBits(Type Ty) const;
};

/// Used to lazily calculate structure layout information for a target machine,
/// based on the DataLayout structure.
class StructLayout final
    : public llvm::TrailingObjects<StructLayout, llvm::TypeSize> {
  llvm::TypeSize StructSize;
  Align StructAlignment;
  unsigned IsPadded : 1;
  unsigned NumElements : 31;

public:
  llvm::TypeSize getSizeInBytes() const { return StructSize; }

  llvm::TypeSize getSizeInBits() const { return 8 * StructSize; }

  Align getAlignment() const { return StructAlignment; }

  /// Returns whether the struct has padding or not between its fields.
  /// NB: Padding in nested element is not taken into account.
  bool hasPadding() const { return IsPadded; }

  /// Given a valid byte offset into the structure, returns the structure
  /// index that contains it.
  unsigned getElementContainingOffset(uint64_t FixedOffset) const;

  MutableArrayRef<llvm::TypeSize> getMemberOffsets() {
    return llvm::MutableArrayRef(getTrailingObjects<llvm::TypeSize>(),
                                 NumElements);
  }

  ArrayRef<llvm::TypeSize> getMemberOffsets() const {
    return llvm::ArrayRef(getTrailingObjects<llvm::TypeSize>(), NumElements);
  }

  llvm::TypeSize getElementOffset(unsigned Idx) const {
    assert(Idx < NumElements && "Invalid element idx!");
    return getMemberOffsets()[Idx];
  }

  llvm::TypeSize getElementOffsetInBits(unsigned Idx) const {
    return getElementOffset(Idx) * 8;
  }

private:
  friend class CIRDataLayout; // Only DataLayout can create this class

  StructLayout(StructType ST, const CIRDataLayout &DL);

  size_t numTrailingObjects(OverloadToken<llvm::TypeSize>) const {
    return NumElements;
  }
};

} // end namespace cir
} // end namespace mlir
