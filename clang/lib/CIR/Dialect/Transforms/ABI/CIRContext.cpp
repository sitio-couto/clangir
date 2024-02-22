#include "CIRContext.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

CIRContext::CIRContext() {}

CIRContext::~CIRContext() {}

TypeInfo CIRContext::getTypeInfo(Type T) const {
  // TODO(cir): Memoize type info.

  TypeInfo TI = getTypeInfoImpl(T);
  return TI;
}

/// getTypeInfoImpl - Return the size of the specified type, in bits.  This
/// method does not work on incomplete types.
///
/// FIXME: Pointers into different addr spaces could have different sizes and
/// alignment requirements: getPointerInfo should take an AddrSpace, this
/// should take a QualType, &c.
TypeInfo CIRContext::getTypeInfoImpl(const Type T) const {
  uint64_t Width = 0;
  unsigned Align = 8;
  AlignRequirementKind AlignRequirement = AlignRequirementKind::None;

  // TODO(cir): We should implement a better way to identify type kinds.
  assert(T.isa<IntType>() && "Unimplemented type class");
  auto typeKind = clang::Type::Builtin;

  // FIXME(cir): Here we fetch the width and alignment of a type considering the
  // current target. We can likely improve this using MLIR's data layout, or
  // some other interface, to abstract this away (e.g. type.getWidth() &
  // type.getAlign()). I'm not sure if data layoot suffices because this would
  // involve some other types such as vectors and complex numbers.
  switch (typeKind) {
  case clang::Type::Builtin: {
    if (auto intTy = T.dyn_cast<IntType>()) {
      Width = Target->getIntWidth();
      Align = Target->getIntAlign();
      break;
    } else {
      llvm_unreachable("Unknown builtin type!");
    }
    break;
  default:
    llvm_unreachable("Unhandled type class");
  }
  }

  assert(llvm::isPowerOf2_32(Align) && "Alignment must be power of 2");
  return TypeInfo(Width, Align, AlignRequirement);
}

void CIRContext::initBuiltinTypes(const clang::TargetInfo &Target,
                                  const clang::TargetInfo *AuxTarget) {
  assert((!this->Target || this->Target == &Target) &&
         "Incorrect target reinitialization");
  this->Target = &Target;
  this->AuxTarget = AuxTarget;
}

/// Get or compute information about the layout of the specified record
/// (struct/union/class), which indicates its size and field position
/// information.
const CIRRecordLayout &CIRContext::getCIRRecordLayout(const Type D) const {
  llvm_unreachable("NYI");
}

} // namespace cir
} // namespace mlir
