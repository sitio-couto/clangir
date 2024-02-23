#include "CIRRecordLayout.h"
#include "CIRContext.h"
#include "MissingFeature.h"

namespace mlir {
namespace cir {

// FIXME(cir): Dummy builder for now.
CIRRecordLayout::CIRRecordLayout() {}

// Constructor for C++ records.
CIRRecordLayout::CIRRecordLayout(
    const CIRContext &Ctx, clang::CharUnits size, clang::CharUnits alignment,
    clang::CharUnits preferredAlignment, clang::CharUnits unadjustedAlignment,
    clang::CharUnits requiredAlignment, bool hasOwnVFPtr,
    bool hasExtendableVFPtr, clang::CharUnits vbptroffset,
    clang::CharUnits datasize, ArrayRef<uint64_t> fieldoffsets,
    clang::CharUnits nonvirtualsize, clang::CharUnits nonvirtualalignment,
    clang::CharUnits preferrednvalignment,
    clang::CharUnits SizeOfLargestEmptySubobject, const Type PrimaryBase,
    bool IsPrimaryBaseVirtual, const Type BaseSharingVBPtr,
    bool EndsWithZeroSizedObject, bool LeadsWithZeroSizedBase)
    : Size(size), DataSize(datasize), Alignment(alignment),
      PreferredAlignment(preferredAlignment),
      UnadjustedAlignment(unadjustedAlignment),
      RequiredAlignment(requiredAlignment), CXXInfo(new CXXRecordLayoutInfo) {
  // NOTE(cir): Clang does a far more elaborate append here by leveraging the
  // custom ASTVector class. For now, we'll do a simple append.
  FieldOffsets.insert(FieldOffsets.end(), fieldoffsets.begin(),
                      fieldoffsets.end());

  assert(!PrimaryBase && "Layout for class with inheritance is NYI");
  // CXXInfo->PrimaryBase.setPointer(PrimaryBase);
  assert(!IsPrimaryBaseVirtual && "Layout for virtual base class is NYI");
  // CXXInfo->PrimaryBase.setInt(IsPrimaryBaseVirtual);
  CXXInfo->NonVirtualSize = nonvirtualsize;
  CXXInfo->NonVirtualAlignment = nonvirtualalignment;
  CXXInfo->PreferredNVAlignment = preferrednvalignment;
  CXXInfo->SizeOfLargestEmptySubobject = SizeOfLargestEmptySubobject;
  // FIXME(cir): I'm assuming that since we are not dealing with inherited
  // classes yet, removing the following lines will be ok.
  // CXXInfo->BaseOffsets = BaseOffsets;
  // CXXInfo->VBaseOffsets = VBaseOffsets;
  CXXInfo->HasOwnVFPtr = hasOwnVFPtr;
  CXXInfo->VBPtrOffset = vbptroffset;
  CXXInfo->HasExtendableVFPtr = hasExtendableVFPtr;
  // FIXME(cir): Probably not necessary for now.
  // CXXInfo->BaseSharingVBPtr = BaseSharingVBPtr;
  CXXInfo->EndsWithZeroSizedObject = EndsWithZeroSizedObject;
  CXXInfo->LeadsWithZeroSizedBase = LeadsWithZeroSizedBase;
}

} // namespace cir
} // namespace mlir
