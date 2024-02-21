#pragma once

namespace mlir {
namespace cir {

struct MissingFeature {

  // Clang CodeGenTypes tracks the set of function being processed. This
  // aparently improves the generated code. However, I'm not sure if we need
  // this feature here since CIR's codegen is already over.
  static bool recursiveFunctionProcessing() { return true; }

  // CIR does not have enough information to easily distinguish different kinds
  // of functions (ctor, dtor, method, etc).
  static bool isCtorOrDtor() { return true; }
  static bool isMethod() { return true; }
  // NOTE(cir): This might not be necessary, since Clang queries Enums to find
  // their underlying integer type, which is already an int in CIR.
  static bool isEnum() { return true; }
  static bool isBuiltinType() { return true; }

  // Some other possible source languages are not yet handled by CIR.
  static bool CUDA() { return true; }
  static bool Swift() { return true; }

  // CIR does not yet hold any form of qualified types. This information is used
  // for ABI lowering and is stripped from the IR until only the canonical type
  // is left. We need to think about how to handle this.
  static bool qualifiedTypes() { return true; }

  // FunctionInfo objects are cached using a profile. This is not yet
  // implemented in CIR, and I'm not sure if it need to be.
  static bool fnInfoProfile() { return true; }

  // Features that will eventually be implemented.
  static bool isCoerceAndExpand() { return true; }
  static bool sretArgument() { return true; }
  static bool inallocaArgument() { return true; }
  static bool argumentPadding() { return true; }
  static bool canBeFlattened() { return true; }
  static bool variadicFunctions() { return true; }
  static bool extParamInfo() { return true; }
  static bool regCall() { return true; }
  static bool chainCall() { return true; }
  static bool vector() { return true; }
  static bool langOpts() { return true; }
  static bool promotableForABI() { return true; }
};

} // namespace cir
} // namespace mlir
