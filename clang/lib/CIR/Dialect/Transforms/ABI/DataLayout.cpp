#include "DataLayout.h"
#include "MissingFeature.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(StructType ST, const CIRDataLayout &DL)
    : StructSize(llvm::TypeSize::getFixed(0)) {
  assert(!ST.isIncomplete() && "Cannot get layout of opaque structs");
  IsPadded = false;
  NumElements = ST.getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    Type Ty = ST.getMembers()[i];
    if (i == 0 && !MissingFeature::isScalableType())
      llvm_unreachable("Scalable types are not yet supported in CIR");

    assert(MissingFeature::packedAttr() && "Cannot identify packed structs");
    const Align TyAlign = DL.getABITypeAlign(Ty);

    // Add padding if necessary to align the data element properly.
    // Currently the only structure with scalable size will be the homogeneous
    // scalable vector types. Homogeneous scalable vector types have members of
    // the same data type so no alignment issue will happen. The condition here
    // assumes so and needs to be adjusted if this assumption changes (e.g. we
    // support structures with arbitrary scalable data type, or structure that
    // contains both fixed size and scalable size data type members).
    if (!StructSize.isScalable() && !isAligned(TyAlign, StructSize)) {
      IsPadded = true;
      StructSize = llvm::TypeSize::getFixed(alignTo(StructSize, TyAlign));
    }

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    getMemberOffsets()[i] = StructSize;
    // Consume space for this data item
    StructSize += DL.getTypeAllocSize(Ty);
  }

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!StructSize.isScalable() && !isAligned(StructAlignment, StructSize)) {
    IsPadded = true;
    StructSize = llvm::TypeSize::getFixed(alignTo(StructSize, StructAlignment));
  }
}

/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t FixedOffset) const {
  assert(!StructSize.isScalable() &&
         "Cannot get element at offset for structure containing scalable "
         "vector types");
  llvm::TypeSize Offset = llvm::TypeSize::getFixed(FixedOffset);
  ArrayRef<llvm::TypeSize> MemberOffsets = getMemberOffsets();

  const auto *SI =
      std::upper_bound(MemberOffsets.begin(), MemberOffsets.end(), Offset,
                       [](llvm::TypeSize LHS, llvm::TypeSize RHS) -> bool {
                         return llvm::TypeSize::isKnownLT(LHS, RHS);
                       });
  assert(SI != MemberOffsets.begin() && "Offset not in structure type!");
  --SI;
  assert(llvm::TypeSize::isKnownLE(*SI, Offset) && "upper_bound didn't work");
  assert((SI == MemberOffsets.begin() ||
          llvm::TypeSize::isKnownLE(*(SI - 1), Offset)) &&
         (SI + 1 == MemberOffsets.end() ||
          llvm::TypeSize::isKnownGT(*(SI + 1), Offset)) &&
         "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI - MemberOffsets.begin();
}

//===----------------------------------------------------------------------===//
// LayoutAlignElem, LayoutAlign support
//===----------------------------------------------------------------------===//

LayoutAlignElem LayoutAlignElem::get(Align ABIAlign, Align PrefAlign,
                                     uint32_t BitWidth) {
  assert(ABIAlign <= PrefAlign && "Preferred alignment worse than ABI!");
  LayoutAlignElem retval;
  retval.ABIAlign = ABIAlign;
  retval.PrefAlign = PrefAlign;
  retval.TypeBitWidth = BitWidth;
  return retval;
}

bool LayoutAlignElem::operator==(const LayoutAlignElem &rhs) const {
  return ABIAlign == rhs.ABIAlign && PrefAlign == rhs.PrefAlign &&
         TypeBitWidth == rhs.TypeBitWidth;
}

//===----------------------------------------------------------------------===//
// PointerAlignElem, PointerAlign support
//===----------------------------------------------------------------------===//

PointerAlignElem PointerAlignElem::getInBits(uint32_t AddressSpace,
                                             Align ABIAlign, Align PrefAlign,
                                             uint32_t TypeBitWidth,
                                             uint32_t IndexBitWidth) {
  assert(ABIAlign <= PrefAlign && "Preferred alignment worse than ABI!");
  PointerAlignElem retval;
  retval.AddressSpace = AddressSpace;
  retval.ABIAlign = ABIAlign;
  retval.PrefAlign = PrefAlign;
  retval.TypeBitWidth = TypeBitWidth;
  retval.IndexBitWidth = IndexBitWidth;
  return retval;
}

bool PointerAlignElem::operator==(const PointerAlignElem &rhs) const {
  return (ABIAlign == rhs.ABIAlign && AddressSpace == rhs.AddressSpace &&
          PrefAlign == rhs.PrefAlign && TypeBitWidth == rhs.TypeBitWidth &&
          IndexBitWidth == rhs.IndexBitWidth);
}

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

namespace {

class StructLayoutMap {
  using LayoutInfoTy = DenseMap<StructType, StructLayout *>;
  LayoutInfoTy LayoutInfo;

public:
  ~StructLayoutMap() {
    // Remove any layouts.
    for (const auto &I : LayoutInfo) {
      StructLayout *Value = I.second;
      Value->~StructLayout();
      free(Value);
    }
  }

  StructLayout *&operator[](StructType STy) { return LayoutInfo[STy]; }
};

} // end anonymous namespace

static const std::pair<AlignTypeEnum, LayoutAlignElem> DefaultAlignments[] = {
    {INTEGER_ALIGN, {1, Align(1), Align(1)}},    // i1
    {INTEGER_ALIGN, {8, Align(1), Align(1)}},    // i8
    {INTEGER_ALIGN, {16, Align(2), Align(2)}},   // i16
    {INTEGER_ALIGN, {32, Align(4), Align(4)}},   // i32
    {INTEGER_ALIGN, {64, Align(4), Align(8)}},   // i64
    {FLOAT_ALIGN, {16, Align(2), Align(2)}},     // half, bfloat
    {FLOAT_ALIGN, {32, Align(4), Align(4)}},     // float
    {FLOAT_ALIGN, {64, Align(8), Align(8)}},     // double
    {FLOAT_ALIGN, {128, Align(16), Align(16)}},  // ppcf128, quad, ...
    {VECTOR_ALIGN, {64, Align(8), Align(8)}},    // v2i32, v1i64, ...
    {VECTOR_ALIGN, {128, Align(16), Align(16)}}, // v16i8, v8i16, v4i32, ...
};

void CIRDataLayout::reset(StringRef Desc) {
  clear();

  LayoutMap = nullptr;
  BigEndian = false;
  AllocaAddrSpace = 0;
  StackNaturalAlign.reset();
  ProgramAddrSpace = 0;
  DefaultGlobalsAddrSpace = 0;
  FunctionPtrAlign.reset();
  TheFunctionPtrAlignType = FunctionPtrAlignType::Independent;
  // ManglingMode = MM_None;
  // NonIntegralAddressSpaces.clear();
  StructAlignment = LayoutAlignElem::get(Align(1), Align(8), 0);

  // Default alignments
  for (const auto &[Kind, Layout] : DefaultAlignments) {
    if (Error Err = setAlignment(Kind, Layout.ABIAlign, Layout.PrefAlign,
                                 Layout.TypeBitWidth))
      return report_fatal_error(std::move(Err));
  }
  if (Error Err = setPointerAlignmentInBits(0, Align(8), Align(8), 64, 64))
    return report_fatal_error(std::move(Err));

  if (Error Err = parseSpecifier(Desc))
    return report_fatal_error(std::move(Err));
}

void CIRDataLayout::clear() {
  LegalIntWidths.clear();
  IntAlignments.clear();
  FloatAlignments.clear();
  VectorAlignments.clear();
  Pointers.clear();
  delete static_cast<StructLayoutMap *>(LayoutMap);
  LayoutMap = nullptr;
}

const StructLayout *CIRDataLayout::getStructLayout(StructType Ty) const {
  if (!LayoutMap)
    LayoutMap = new StructLayoutMap();

  StructLayoutMap *STM = static_cast<StructLayoutMap *>(LayoutMap);
  StructLayout *&SL = (*STM)[Ty];
  if (SL)
    return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we
  // malloc it, then use placement new.
  StructLayout *L = (StructLayout *)llvm::safe_malloc(
      StructLayout::totalSizeToAlloc<llvm::TypeSize>(Ty.getNumElements()));

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) StructLayout(Ty, *this);

  return L;
}

Error CIRDataLayout::setAlignment(AlignTypeEnum AlignType, Align ABIAlign,
                                  Align PrefAlign, uint32_t BitWidth) {
  // AlignmentsTy::ABIAlign and AlignmentsTy::PrefAlign were once stored as
  // uint16_t, it is unclear if there are requirements for alignment to be less
  // than 2^16 other than storage. In the meantime we leave the restriction as
  // an assert. See D67400 for context.
  assert(Log2(ABIAlign) < 16 && Log2(PrefAlign) < 16 && "Alignment too big");
  if (!llvm::isUInt<24>(BitWidth))
    llvm_unreachable("Invalid bit width, must be a 24-bit integer");
  if (PrefAlign < ABIAlign)
    llvm_unreachable(
        "Preferred alignment cannot be less than the ABI alignment");

  SmallVectorImpl<LayoutAlignElem> *Alignments;
  switch (AlignType) {
  case AGGREGATE_ALIGN:
    StructAlignment.ABIAlign = ABIAlign;
    StructAlignment.PrefAlign = PrefAlign;
    return Error::success();
  case INTEGER_ALIGN:
    Alignments = &IntAlignments;
    break;
  case FLOAT_ALIGN:
    Alignments = &FloatAlignments;
    break;
  case VECTOR_ALIGN:
    Alignments = &VectorAlignments;
    break;
  }

  auto *I = partition_point(*Alignments, [BitWidth](const LayoutAlignElem &E) {
    return E.TypeBitWidth < BitWidth;
  });
  if (I != Alignments->end() && I->TypeBitWidth == BitWidth) {
    // Update the abi, preferred alignments.
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
  } else {
    // Insert before I to keep the vector sorted.
    Alignments->insert(I, LayoutAlignElem::get(ABIAlign, PrefAlign, BitWidth));
  }
  return Error::success();
}

Error CIRDataLayout::setPointerAlignmentInBits(uint32_t AddrSpace,
                                               Align ABIAlign, Align PrefAlign,
                                               uint32_t TypeBitWidth,
                                               uint32_t IndexBitWidth) {
  if (PrefAlign < ABIAlign)
    llvm_unreachable(
        "Preferred alignment cannot be less than the ABI alignment");
  if (IndexBitWidth > TypeBitWidth)
    llvm_unreachable("Index width cannot be larger than pointer width");

  assert(MissingFeature::addresSpace() &&
         "Address space is not yet supported in CIR");
  auto *I = lower_bound(Pointers, 0,
                        [](const PointerAlignElem &A, uint32_t AddressSpace) {
                          return A.AddressSpace < AddressSpace;
                        });
  if (I == Pointers.end() || I->AddressSpace != AddrSpace) {
    Pointers.insert(I,
                    PointerAlignElem::getInBits(AddrSpace, ABIAlign, PrefAlign,
                                                TypeBitWidth, IndexBitWidth));
  } else {
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
    I->TypeBitWidth = TypeBitWidth;
    I->IndexBitWidth = IndexBitWidth;
  }
  return Error::success();
}

/// Get an unsigned integer, including error checks.
template <typename IntTy> static Error getInt(StringRef R, IntTy &Result) {
  bool error = R.getAsInteger(10, Result);
  (void)error;
  if (error)
    llvm_unreachable("not a number, or does not fit in an unsigned int");
  return Error::success();
}

/// Checked version of split, to ensure mandatory subparts.
static Error split(StringRef Str, char Separator,
                   std::pair<StringRef, StringRef> &Split) {
  assert(!Str.empty() && "parse error, string can't be empty here");
  Split = Str.split(Separator);
  if (Split.second.empty() && Split.first != Str)
    llvm_unreachable("Trailing separator in datalayout string");
  if (!Split.second.empty() && Split.first.empty())
    llvm_unreachable("Expected token before separator in datalayout string");
  return Error::success();
}

/// Get an unsigned integer representing the number of bits and convert it into
/// bytes. Error out of not a byte width multiple.
template <typename IntTy>
static Error getIntInBytes(StringRef R, IntTy &Result) {
  if (Error Err = getInt<IntTy>(R, Result))
    return Err;
  if (Result % 8)
    llvm_unreachable("number of bits must be a byte width multiple");
  Result /= 8;
  return Error::success();
}

Error CIRDataLayout::parseSpecifier(StringRef Desc) {
  StringRepresentation = std::string(Desc);
  while (!Desc.empty()) {
    // Split at '-'.
    std::pair<StringRef, StringRef> Split;
    if (Error Err = split(Desc, '-', Split))
      return Err;
    Desc = Split.second;

    // Split at ':'.
    if (Error Err = split(Split.first, ':', Split))
      return Err;

    // Aliases used below.
    StringRef &Tok = Split.first;   // Current token.
    StringRef &Rest = Split.second; // The rest of the string.

    if (Tok == "ni") {
      do {
        if (Error Err = split(Rest, ':', Split))
          return Err;
        Rest = Split.second;
        unsigned AS;
        if (Error Err = getInt(Split.first, AS))
          return Err;
        if (AS == 0)
          llvm_unreachable("Address space 0 can never be non-integral");
        llvm_unreachable("Non-integral address spaces is NYI");
      } while (!Rest.empty());

      continue;
    }

    char Specifier = Tok.front();
    Tok = Tok.substr(1);

    switch (Specifier) {
    case 's':
      // Deprecated, but ignoring here to preserve loading older textual llvm
      // ASM file
      break;
    case 'E':
      BigEndian = true;
      break;
    case 'e':
      BigEndian = false;
      break;
    case 'p': {
      // Address space.
      unsigned AddrSpace = 0;
      if (!Tok.empty())
        if (Error Err = getInt(Tok, AddrSpace))
          return Err;
      if (!llvm::isUInt<24>(AddrSpace))
        llvm_unreachable("Invalid address space, must be a 24-bit integer");

      // Size.
      if (Rest.empty())
        llvm_unreachable(
            "Missing size specification for pointer in datalayout string");
      if (Error Err = split(Rest, ':', Split))
        return Err;
      unsigned PointerMemSize;
      if (Error Err = getInt(Tok, PointerMemSize))
        return Err;
      if (!PointerMemSize)
        llvm_unreachable("Invalid pointer size of 0 bytes");

      // ABI alignment.
      if (Rest.empty())
        llvm_unreachable(
            "Missing alignment specification for pointer in datalayout string");
      if (Error Err = split(Rest, ':', Split))
        return Err;
      unsigned PointerABIAlign;
      if (Error Err = getIntInBytes(Tok, PointerABIAlign))
        return Err;
      if (!llvm::isPowerOf2_64(PointerABIAlign))
        llvm_unreachable("Pointer ABI alignment must be a power of 2");

      // Size of index used in GEP for address calculation.
      // The parameter is optional. By default it is equal to size of pointer.
      unsigned IndexSize = PointerMemSize;

      // Preferred alignment.
      unsigned PointerPrefAlign = PointerABIAlign;
      if (!Rest.empty()) {
        if (Error Err = split(Rest, ':', Split))
          return Err;
        if (Error Err = getIntInBytes(Tok, PointerPrefAlign))
          return Err;
        if (!llvm::isPowerOf2_64(PointerPrefAlign))
          llvm_unreachable("Pointer preferred alignment must be a power of 2");

        // Now read the index. It is the second optional parameter here.
        if (!Rest.empty()) {
          if (Error Err = split(Rest, ':', Split))
            return Err;
          if (Error Err = getInt(Tok, IndexSize))
            return Err;
          if (!IndexSize)
            llvm_unreachable("Invalid index size of 0 bytes");
        }
      }
      if (Error Err = setPointerAlignmentInBits(
              AddrSpace, llvm::assumeAligned(PointerABIAlign),
              llvm::assumeAligned(PointerPrefAlign), PointerMemSize, IndexSize))
        return Err;
      break;
    }
    case 'i':
    case 'v':
    case 'f':
    case 'a': {
      AlignTypeEnum AlignType;
      switch (Specifier) {
      default:
        llvm_unreachable("Unexpected specifier!");
      case 'i':
        AlignType = INTEGER_ALIGN;
        break;
      case 'v':
        AlignType = VECTOR_ALIGN;
        break;
      case 'f':
        AlignType = FLOAT_ALIGN;
        break;
      case 'a':
        AlignType = AGGREGATE_ALIGN;
        break;
      }

      // Bit size.
      unsigned Size = 0;
      if (!Tok.empty())
        if (Error Err = getInt(Tok, Size))
          return Err;

      if (AlignType == AGGREGATE_ALIGN && Size != 0)
        llvm_unreachable("Sized aggregate specification in datalayout string");

      // ABI alignment.
      if (Rest.empty())
        llvm_unreachable(
            "Missing alignment specification in datalayout string");
      if (Error Err = split(Rest, ':', Split))
        return Err;
      unsigned ABIAlign;
      if (Error Err = getIntInBytes(Tok, ABIAlign))
        return Err;
      if (AlignType != AGGREGATE_ALIGN && !ABIAlign)
        llvm_unreachable(
            "ABI alignment specification must be >0 for non-aggregate types");

      if (!llvm::isUInt<16>(ABIAlign))
        llvm_unreachable("Invalid ABI alignment, must be a 16bit integer");
      if (ABIAlign != 0 && !llvm::isPowerOf2_64(ABIAlign))
        llvm_unreachable("Invalid ABI alignment, must be a power of 2");
      if (AlignType == INTEGER_ALIGN && Size == 8 && ABIAlign != 1)
        llvm_unreachable("Invalid ABI alignment, i8 must be naturally aligned");

      // Preferred alignment.
      unsigned PrefAlign = ABIAlign;
      if (!Rest.empty()) {
        if (Error Err = split(Rest, ':', Split))
          return Err;
        if (Error Err = getIntInBytes(Tok, PrefAlign))
          return Err;
      }

      if (!llvm::isUInt<16>(PrefAlign))
        llvm_unreachable(
            "Invalid preferred alignment, must be a 16bit integer");
      if (PrefAlign != 0 && !llvm::isPowerOf2_64(PrefAlign))
        llvm_unreachable("Invalid preferred alignment, must be a power of 2");

      if (Error Err = setAlignment(AlignType, llvm::assumeAligned(ABIAlign),
                                   llvm::assumeAligned(PrefAlign), Size))
        return Err;

      break;
    }
    case 'n': // Native integer types.
      while (true) {
        unsigned Width;
        if (Error Err = getInt(Tok, Width))
          return Err;
        if (Width == 0)
          llvm_unreachable(
              "Zero width native integer type in datalayout string");
        LegalIntWidths.push_back(Width);
        if (Rest.empty())
          break;
        if (Error Err = split(Rest, ':', Split))
          return Err;
      }
      break;
    case 'S': { // Stack natural alignment.
      uint64_t Alignment;
      if (Error Err = getIntInBytes(Tok, Alignment))
        return Err;
      if (Alignment != 0 && !llvm::isPowerOf2_64(Alignment))
        llvm_unreachable("Alignment is neither 0 nor a power of 2");
      StackNaturalAlign = llvm::MaybeAlign(Alignment);
      break;
    }
    case 'F': {
      switch (Tok.front()) {
      case 'i':
        TheFunctionPtrAlignType = FunctionPtrAlignType::Independent;
        break;
      case 'n':
        TheFunctionPtrAlignType = FunctionPtrAlignType::MultipleOfFunctionAlign;
        break;
      default:
        llvm_unreachable("Unknown function pointer alignment type in "
                         "datalayout string");
      }
      Tok = Tok.substr(1);
      uint64_t Alignment;
      if (Error Err = getIntInBytes(Tok, Alignment))
        return Err;
      if (Alignment != 0 && !llvm::isPowerOf2_64(Alignment))
        llvm_unreachable("Alignment is neither 0 nor a power of 2");
      FunctionPtrAlign = llvm::MaybeAlign(Alignment);
      break;
    }
    case 'P': { // Function address space.
      llvm_unreachable("Address space is NYI");
      break;
    }
    case 'A': { // Default stack/alloca address space.
      llvm_unreachable("Address space is NYI");
      break;
    }
    case 'G': { // Default address space for global variables.
      llvm_unreachable("Address space is NYI");
      break;
    }
    case 'm':
      if (!Tok.empty())
        llvm_unreachable("Unexpected trailing characters after mangling "
                         "specifier in datalayout string");
      if (Rest.empty())
        llvm_unreachable("Expected mangling specifier in datalayout string");
      if (Rest.size() > 1)
        llvm_unreachable("Unknown mangling specifier in datalayout string");
      switch (Rest[0]) {
      default:
        llvm_unreachable("Unknown mangling in datalayout string");
      case 'e':
        ManglingMode = MM_ELF;
        break;
      case 'l':
        ManglingMode = MM_GOFF;
        break;
      case 'o':
        ManglingMode = MM_MachO;
        break;
      case 'm':
        ManglingMode = MM_Mips;
        break;
      case 'w':
        ManglingMode = MM_WinCOFF;
        break;
      case 'x':
        ManglingMode = MM_WinCOFFX86;
        break;
      case 'a':
        ManglingMode = MM_XCOFF;
        break;
      }
      break;
    default:
      llvm_unreachable("Unknown specifier in datalayout string");
      break;
    }
  }

  return Error::success();
}

static SmallVectorImpl<LayoutAlignElem>::const_iterator
findAlignmentLowerBound(const SmallVectorImpl<LayoutAlignElem> &Alignments,
                        uint32_t BitWidth) {
  return partition_point(Alignments, [BitWidth](const LayoutAlignElem &E) {
    return E.TypeBitWidth < BitWidth;
  });
}

Align CIRDataLayout::getIntegerAlignment(uint32_t BitWidth,
                                         bool abi_or_pref) const {
  const auto *I = findAlignmentLowerBound(IntAlignments, BitWidth);
  // If we don't have an exact match, use alignment of next larger integer
  // type. If there is none, use alignment of largest integer type by going
  // back one element.
  if (I == IntAlignments.end())
    --I;
  return abi_or_pref ? I->ABIAlign : I->PrefAlign;
}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
Align CIRDataLayout::getAlignment(Type Ty, bool abi_or_pref) const {
  if (auto intTy = Ty.dyn_cast<IntType>()) {
    return getIntegerAlignment(intTy.getWidth(), abi_or_pref);
  }
  if (Ty.isa<StructType>()) {
    // Packed structure types always have an ABI alignment of one.
    if (!MissingFeature::packedAttr() && abi_or_pref)
      llvm_unreachable("NYI");

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    const Align Align =
        abi_or_pref ? StructAlignment.ABIAlign : StructAlignment.PrefAlign;
    return std::max(Align, Layout->getAlignment());
  }
  if (auto PtrTy = Ty.dyn_cast<PointerType>()) {
    // FIXME(cir): This does not account for differnt address spaces, and relies
    // on CIR's data layout to give the proper alignment.
    assert(MissingFeature::addresSpace());
    uint align = abi_or_pref ? DL.getTypeABIAlignment(PtrTy)
                             : DL.getTypePreferredAlignment(PtrTy);
    return llvm::Align(align);
  }
  if (auto floatTy = Ty.dyn_cast<FloatType>()) {
    // FIXME(cir): We should be able to use MLIR's datalayout interface to
    // easily query this.
    unsigned BitWidth = floatTy.getWidth();
    auto *I = findAlignmentLowerBound(FloatAlignments, BitWidth);
    if (I != FloatAlignments.end() && I->TypeBitWidth == BitWidth)
      return abi_or_pref ? I->ABIAlign : I->PrefAlign;
  }
  llvm::errs() << "Type: " << Ty << "\n";
  llvm_unreachable("CIRDataLayout::getAlignment(): Unsupported type");
}

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
inline llvm::TypeSize CIRDataLayout::getTypeSizeInBits(Type Ty) const {
  assert(MissingFeature::isSized() &&
         "Cannot getTypeInfo() on a type that is unsized!");
  if (auto intTy = Ty.dyn_cast<IntType>())
    return llvm::TypeSize::getFixed(intTy.getWidth());
  if (auto structTy = Ty.dyn_cast<StructType>()) {
    // Get the layout annotation... which is lazily created on demand.
    return getStructLayout(structTy)->getSizeInBits();
  }
  if (auto PtrTy = Ty.dyn_cast<PointerType>()) {
    // FIXME(cir): This does not account for differnt address spaces, and relies
    // on CIR's data layout to give the proper ABI-specific type width.
    assert(MissingFeature::addresSpace());
    return llvm::TypeSize::getFixed(DL.getTypeSizeInBits(PtrTy));
  }
  if (auto floatTy = Ty.dyn_cast<FloatType>()) {
    return llvm::TypeSize::getFixed(floatTy.getWidth());
  }
  llvm::errs() << "Type: " << Ty << "\n";
  llvm_unreachable("CIRDataLayout::getTypeSizeInBits(): Unsupported type");
}

} // end namespace cir
} // end namespace mlir
