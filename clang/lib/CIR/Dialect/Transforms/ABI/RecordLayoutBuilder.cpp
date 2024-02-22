#include "CIRContext.h"
#include "MissingFeature.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace mlir;
using namespace cir;

namespace {

/// Keeps track of which empty subobjects exist at different offsets while
/// laying out a C++ class.
class EmptySubobjectMap {};

class ItaniumRecordLayoutBuilder {};

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
    assert(MissingFeature::isCXXRecord());
    llvm_unreachable("NYI");
  }
}
