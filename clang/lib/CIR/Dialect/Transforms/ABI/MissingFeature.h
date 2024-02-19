#pragma once

namespace mlir {
namespace cir {

struct MissingFeature {

  // Clang CodeGenTypes tracks the set of function being processed. This
  // aparently improves the generated code. However, I'm not sure if we need
  // this feature here since CIR's codegen is already over.
  static bool recursiveFunctionProcessing() { return true; }

  // Features that will eventually be implemented.
  static bool isCoerceAndExpand() { return true; }
  static bool sretArgument() { return true; }
  static bool inallocaArgument() { return true; }
  static bool argumentPadding() { return true; }
  static bool canBeFlattened() { return true; }
  static bool variadicFunctions() { return true; }
};

} // namespace cir
} // namespace mlir
