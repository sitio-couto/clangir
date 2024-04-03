#include "CIRContext.h"
#include "CIRRecordLayout.h"
#include "MissingFeature.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>

namespace mlir {
namespace cir {

CIRContext::CIRContext(MLIRContext *MLIRCtx, clang::LangOptions &LOpts)
    : MLIRCtx(MLIRCtx), LangOpts(LOpts) {}

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

  // TODO(cir): We should implement a better way to identify type kinds and use
  // builting data layout interface for this.
  auto typeKind = clang::Type::Builtin;
  if (T.isa<IntType, Float32Type, Float64Type, BoolType>()) {
    typeKind = clang::Type::Builtin;
  } else if (T.isa<StructType>()) {
    typeKind = clang::Type::Record;
  } else {
    llvm_unreachable("Unhandled type class");
  }

  // FIXME(cir): Here we fetch the width and alignment of a type considering the
  // current target. We can likely improve this using MLIR's data layout, or
  // some other interface, to abstract this away (e.g. type.getWidth() &
  // type.getAlign()). I'm not sure if data layoot suffices because this would
  // involve some other types such as vectors and complex numbers.
  // FIXME(cir): In the original codegen, this receives an AST type, meaning it
  // differs chars from integers, something that is not possible with the
  // current level of CIR.
  switch (typeKind) {
  case clang::Type::Builtin: {
    if (auto boolTy = T.dyn_cast<BoolType>()) {
      Width = Target->getBoolWidth();
      Align = Target->getBoolAlign();
      break;
    }
    if (auto intTy = T.dyn_cast<IntType>()) {
      // NOTE(cir): This assumes int types are already ABI-specific.
      // FIXME(cir): Use data layout interface here instead.
      Width = intTy.getWidth();
      // FIXME(cir): Use the proper getABIAlignment method here.
      Align = std::ceil((float)Width / 8) * 8;
      break;
    }
    if (auto floatTy = T.dyn_cast<Float32Type>()) {
      Width = Target->getFloatWidth();
      Align = Target->getFloatAlign();
      break;
    }
    if (auto doubleTy = T.dyn_cast<Float64Type>()) {
      Width = Target->getDoubleWidth();
      Align = Target->getDoubleAlign();
      break;
    }
    llvm_unreachable("Unknown builtin type!");
    break;
  }
  case clang::Type::Record: {
    const auto RT = T.dyn_cast<StructType>();
    assert(MissingFeature::tagTypeClass());

    // Only handle TagTypes (names types) for now.
    assert(RT.getName() && "Anonymous record is NYI");

    // NOTE(cir): Clang does some hanlding of invalid tagged declarations here.
    // Not sure if this is necessary in CIR.

    if (!MissingFeature::isEnum()) {
      llvm_unreachable("NYI");
    }

    const CIRRecordLayout &Layout = getCIRRecordLayout(RT);
    Width = toBits(Layout.getSize());
    Align = toBits(Layout.getAlignment());
    assert(MissingFeature::alignmentAttribute());
    break;
  }
  default:
    llvm_unreachable("Unhandled type class");
  }

  assert(llvm::isPowerOf2_32(Align) && "Alignment must be power of 2");
  return TypeInfo(Width, Align, AlignRequirement);
}

/// Convert a size in characters to a size in characters.
int64_t CIRContext::toBits(clang::CharUnits CharSize) const {
  return CharSize.getQuantity() * getCharWidth();
}

clang::CharUnits CIRContext::getTypeSizeInChars(Type T) const {
  return getTypeInfoInChars(T).Width;
}

void CIRContext::initBuiltinType(Type &Ty, clang::BuiltinType::Kind K) {
  // NOTE(cir): Clang does more stuff here. Not sure if we need to do the same.
  assert(MissingFeature::qualifiedTypes());
  switch (K) {
  case clang::BuiltinType::Char_S:
    Ty = IntType::get(getMLIRContext(), 8, true);
    break;
  default:
    llvm_unreachable("NYI");
  }
  Types.push_back(Ty);
}

void CIRContext::initBuiltinTypes(const clang::TargetInfo &Target,
                                  const clang::TargetInfo *AuxTarget) {
  assert((!this->Target || this->Target == &Target) &&
         "Incorrect target reinitialization");
  this->Target = &Target;
  this->AuxTarget = AuxTarget;

  // C99 6.2.5p3.
  if (LangOpts.CharIsSigned)
    initBuiltinType(CharTy, clang::BuiltinType::Char_S);
  else
    llvm_unreachable("NYI");
}

/// toCharUnitsFromBits - Convert a size in bits to a size in characters.
clang::CharUnits CIRContext::toCharUnitsFromBits(int64_t BitSize) const {
  return clang::CharUnits::fromQuantity(BitSize / getCharWidth());
}

TypeInfoChars CIRContext::getTypeInfoInChars(Type T) const {
  if (auto arrTy = T.dyn_cast<ArrayType>())
    llvm_unreachable("NYI");
  TypeInfo Info = getTypeInfo(T);
  return TypeInfoChars(toCharUnitsFromBits(Info.Width),
                       toCharUnitsFromBits(Info.Align), Info.AlignRequirement);
}

bool CIRContext::isPromotableIntegerType(Type T) const {
  // HLSL doesn't promote all small integer types to int, it
  // just uses the rank-based promotion rules for all types.
  if (getLangOpts().HLSL)
    llvm_unreachable("NYI");

  // FIXME(cir): We have no way of properly identifying if the integer was
  // originally a char, short, etc. So we just assume it is promotable if
  // smaller than 32 bits. This is wrong since, for example, Char32 is
  // promotable.
  assert(MissingFeature::isBuiltinType());
  if (T.isa<BoolType>() ||
      (T.isa<IntType>() && T.cast<IntType>().getWidth() < 32)) {
    return true;
  }
  if (T.isa<IntType>() && T.cast<IntType>().getWidth() >= 32) {
    return false;
  }

  // Enumerated types are promotable to their compatible integer types
  // (C99 6.3.1.1) a.k.a. its underlying type (C++ [conv.prom]p2).
  // TODO(cir): We have no way of knowing if a integer originated from an enum.
  if (MissingFeature::isEnum()) {
    llvm_unreachable("NYI");
  }

  return false;
}

} // namespace cir
} // namespace mlir
