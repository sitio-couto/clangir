#include "CIRContext.h"
#include "MissingFeature.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace cir;

namespace {

//===-----------------------------------------------------------------------==//
// EmptySubobjectMap Implementation
//===----------------------------------------------------------------------===//

/// Keeps track of which empty subobjects exist at different offsets while
/// laying out a C++ class.
class EmptySubobjectMap {
  const CIRContext &Context;
  uint64_t CharWidth;

  /// The class whose empty entries we're keeping track of.
  const StructType Class;

  /// The highest offset known to contain an empty base subobject.
  clang::CharUnits MaxEmptyClassOffset;

  /// Compute the size of the largest base or member subobject that is empty.
  void ComputeEmptySubobjectSizes();

public:
  EmptySubobjectMap(const CIRContext &Context, const StructType Class)
      : Context(Context), CharWidth(Context.getCharWidth()), Class(Class) {
    ComputeEmptySubobjectSizes();
  }
};

void EmptySubobjectMap::ComputeEmptySubobjectSizes() {
  // Check the bases.
  assert(MissingFeature::recordBasesIterator());

  // Check the fields.
  for (const auto FT : Class.getMembers()) {
    assert(MissingFeature::qualifiedTypes());
    const auto RT = FT.dyn_cast<StructType>();

    // We only care about record types.
    if (!RT)
      continue;

    // TODO(cir): Handle nested record types.
    llvm_unreachable("NYI");
  }
}

//===-----------------------------------------------------------------------==//
// ItaniumRecordLayoutBuilder Implementation
//===----------------------------------------------------------------------===//

class ItaniumRecordLayoutBuilder {
protected:
  // FIXME(cir):  Remove this and make the appropriate fields public.
  friend class CIRContext;

  const CIRContext &Context;

  EmptySubobjectMap *EmptySubobjects;

  /// Size - The current size of the record layout.
  uint64_t Size;

  /// Alignment - The current alignment of the record layout.
  clang::CharUnits Alignment;

  /// PreferredAlignment - The preferred alignment of the record layout.
  clang::CharUnits PreferredAlignment;

  /// The alignment if attribute packed is not used.
  clang::CharUnits UnpackedAlignment;

  /// \brief The maximum of the alignments of top-level members.
  clang::CharUnits UnadjustedAlignment;

  SmallVector<uint64_t, 16> FieldOffsets;

  /// Whether the external AST source has provided a layout for this
  /// record.
  unsigned UseExternalLayout : 1;

  /// Whether we need to infer alignment, even when we have an
  /// externally-provided layout.
  unsigned InferAlignment : 1;

  /// Packed - Whether the record is packed or not.
  unsigned Packed : 1;

  unsigned IsUnion : 1;

  unsigned IsMac68kAlign : 1;

  unsigned IsNaturalAlign : 1;

  unsigned IsMsStruct : 1;

  /// UnfilledBitsInLastUnit - If the last field laid out was a bitfield,
  /// this contains the number of bits in the last unit that can be used for
  /// an adjacent bitfield if necessary.  The unit in question is usually
  /// a byte, but larger units are used if IsMsStruct.
  unsigned char UnfilledBitsInLastUnit;

  /// LastBitfieldStorageUnitSize - If IsMsStruct, represents the size of the
  /// storage unit of the previous field if it was a bitfield.
  unsigned char LastBitfieldStorageUnitSize;

  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack.
  clang::CharUnits MaxFieldAlignment;

  /// DataSize - The data size of the record being laid out.
  uint64_t DataSize;

  clang::CharUnits NonVirtualSize;
  clang::CharUnits NonVirtualAlignment;
  clang::CharUnits PreferredNVAlignment;

  /// If we've laid out a field but not included its tail padding in Size yet,
  /// this is the size up to the end of that field.
  clang::CharUnits PaddedFieldSize;

  /// PrimaryBaseIsVirtual - Whether the primary base of the class we're laying
  /// out is virtual.
  bool PrimaryBaseIsVirtual;

  /// Whether the class provides its own vtable/vftbl pointer, as opposed to
  /// inheriting one from a primary base class.
  bool HasOwnVFPtr;

  /// the flag of field offset changing due to packed attribute.
  bool HasPackedField;

  /// An auxiliary field used for AIX. When there are OverlappingEmptyFields
  /// existing in the aggregate, the flag shows if the following first non-empty
  /// or empty-but-non-overlapping field has been handled, if any.
  bool HandledFirstNonOverlappingEmptyField;

public:
  ItaniumRecordLayoutBuilder(const CIRContext &Context,
                             EmptySubobjectMap *EmptySubobjects)
      : Context(Context), EmptySubobjects(EmptySubobjects), Size(0),
        Alignment(clang::CharUnits::One()),
        PreferredAlignment(clang::CharUnits::One()),
        UnpackedAlignment(clang::CharUnits::One()),
        UnadjustedAlignment(clang::CharUnits::One()), UseExternalLayout(false),
        InferAlignment(false), Packed(false), IsUnion(false),
        IsMac68kAlign(false),
        IsNaturalAlign(!Context.getTargetInfo().getTriple().isOSAIX()),
        IsMsStruct(false), UnfilledBitsInLastUnit(0),
        LastBitfieldStorageUnitSize(0),
        MaxFieldAlignment(clang::CharUnits::Zero()), DataSize(0),
        NonVirtualSize(clang::CharUnits::Zero()),
        NonVirtualAlignment(clang::CharUnits::One()),
        PreferredNVAlignment(clang::CharUnits::One()),
        PaddedFieldSize(clang::CharUnits::Zero()), PrimaryBaseIsVirtual(false),
        HasOwnVFPtr(false), HasPackedField(false),
        HandledFirstNonOverlappingEmptyField(false) {}

  void layout(const StructType D);

  /// Initialize record layout for the given record decl.
  void initializeLayout(const Type Ty);
};

void ItaniumRecordLayoutBuilder::layout(const StructType RT) {
  initializeLayout(RT);

  // Lay out the vtable and the non-virtual bases.
  assert(MissingFeature::isCXXRecord() && MissingFeature::isDynamicClass());

  // LayoutFields(RD);

  // NonVirtualSize = Context.toCharUnitsFromBits(
  //     llvm::alignTo(getSizeInBits(),
  //     Context.getTargetInfo().getCharAlign()));
  // NonVirtualAlignment = Alignment;
  // PreferredNVAlignment = PreferredAlignment;

  // // Lay out the virtual bases and add the primary virtual base offsets.
  // LayoutVirtualBases(RD, RD);

  // // Finally, round the size of the total struct up to the alignment
  // // of the struct itself.
  // FinishLayout(RD);
}

void ItaniumRecordLayoutBuilder::initializeLayout(const mlir::Type Ty) {
  if (const auto RT = Ty.dyn_cast<StructType>()) {
    IsUnion = RT.isUnion();
    assert(MissingFeature::MSStructAttr());
  }

  assert(MissingFeature::packedAttr());

  // Honor the default struct packing maximum alignment flag.
  if (unsigned DefaultMaxFieldAlignment = Context.getLangOpts().PackStruct) {
    llvm_unreachable("NYI");
  }

  // mac68k alignment supersedes maximum field alignment and attribute aligned,
  // and forces all structures to have 2-byte alignment. The IBM docs on it
  // allude to additional (more complicated) semantics, especially with regard
  // to bit-fields, but gcc appears not to follow that.
  if (MissingFeature::alignMac68kAttr()) {
    llvm_unreachable("NYI");
  } else {
    if (MissingFeature::alignNaturalAttr())
      llvm_unreachable("NYI");

    if (MissingFeature::maxFieldAlignmentAttr())
      llvm_unreachable("NYI");

    if (MissingFeature::getMaxAlignment())
      llvm_unreachable("NYI");
  }

  HandledFirstNonOverlappingEmptyField =
      !Context.getTargetInfo().defaultsToAIXPowerAlignment() || IsNaturalAlign;

  // If there is an external AST source, ask it for the various offsets.
  if (const auto RT = Ty.dyn_cast<StructType>()) {
    if (MissingFeature::externalASTSource()) {
      llvm_unreachable("NYI");
    }
  }
}

bool isMsLayout(const CIRContext &Context) {
  return Context.getTargetInfo().getCXXABI().isMicrosoft();
}

} // namespace

/// Get or compute information about the layout of the specified record
/// (struct/union/class), which indicates its size and field position
/// information.
const CIRRecordLayout &CIRContext::getCIRRecordLayout(const Type D) const {
  assert(D.isa<StructType>() && "Not a record type");
  auto RT = D.dyn_cast<StructType>();

  assert(RT.isComplete() && "Cannot get layout of forward declarations!");

  // FIXME(cir): Cache the layout. Also, use a more MLIR-based approach.

  const CIRRecordLayout *NewEntry = nullptr;

  if (isMsLayout(*this)) {
    llvm_unreachable("NYI");
  } else {
    // FIXME(cir): Add if-else separating C and C++ records.
    assert(MissingFeature::isCXXRecord());
    EmptySubobjectMap EmptySubobjects(*this, RT);
    ItaniumRecordLayoutBuilder Builder(*this, &EmptySubobjects);
    // Builder.Layout(RD);

    llvm_unreachable("NYI");
  }
}
