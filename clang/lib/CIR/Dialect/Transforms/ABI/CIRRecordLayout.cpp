#include "CIRRecordLayout.h"
#include "CIRContext.h"

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
      RequiredAlignment(requiredAlignment),
      CXXInfo(new CXXRecordLayoutInfo) {
  // FieldOffsets.append(Ctx, fieldoffsets.begin(), fieldoffsets.end());

//   CXXInfo->PrimaryBase.setPointer(PrimaryBase);
//   CXXInfo->PrimaryBase.setInt(IsPrimaryBaseVirtual);
//   CXXInfo->NonVirtualSize = nonvirtualsize;
//   CXXInfo->NonVirtualAlignment = nonvirtualalignment;
//   CXXInfo->PreferredNVAlignment = preferrednvalignment;
//   CXXInfo->SizeOfLargestEmptySubobject = SizeOfLargestEmptySubobject;
//   CXXInfo->BaseOffsets = BaseOffsets;
//   CXXInfo->VBaseOffsets = VBaseOffsets;
//   CXXInfo->HasOwnVFPtr = hasOwnVFPtr;
//   CXXInfo->VBPtrOffset = vbptroffset;
//   CXXInfo->HasExtendableVFPtr = hasExtendableVFPtr;
//   CXXInfo->BaseSharingVBPtr = BaseSharingVBPtr;
//   CXXInfo->EndsWithZeroSizedObject = EndsWithZeroSizedObject;
//   CXXInfo->LeadsWithZeroSizedBase = LeadsWithZeroSizedBase;

// #ifndef NDEBUG
//   if (const CXXRecordDecl *PrimaryBase = getPrimaryBase()) {
//     if (isPrimaryBaseVirtual()) {
//       if (Ctx.getTargetInfo().getCXXABI().hasPrimaryVBases()) {
//         assert(getVBaseClassOffset(PrimaryBase).isZero() &&
//                "Primary virtual base must be at offset 0!");
//       }
//     } else {
//       assert(getBaseClassOffset(PrimaryBase).isZero() &&
//              "Primary base must be at offset 0!");
//     }
//   }
// #endif
}

} // namespace cir
} // namespace mlir
