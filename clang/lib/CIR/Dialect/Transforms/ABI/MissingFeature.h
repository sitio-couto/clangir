#pragma once

namespace mlir {
namespace cir {

struct MissingFeature {

  // CIR does not yet have a concept of language options. This is used to
  // control the behavior of the compiler in various areas. When dealing with
  // ABI, some options might affect the behaviour.
  static bool langOptions() { return true; }

  // Clang CodeGenTypes tracks the set of function being processed. This
  // aparently improves the generated code. However, I'm not sure if we need
  // this feature here since CIR's codegen is already over.
  static bool recursiveFunctionProcessing() { return true; }

  // Clang uses several abstractions to facilitate handling of types. A few
  // examples are QualType, TagType, CXXRecordDecl, etc. CIR should have its
  // own abstractions mirroring these. It would facilitate codegen parity.
  static bool tagTypeClass() { return true; }

  // Clang has a class that abstracts external AST sources. Not sure if this
  // will be necessary for CIR.
  static bool externalASTSource() { return true; }

  // ABI codegen has several diagnostics that are not yet implemented in CIR.
  // These involve several warning messages which might not make sense in CIR.
  static bool diagnostics() { return true; }

  // CIR does not have enough information to easily distinguish certain
  // properties between language elements. For example, it can't distinguish
  // functions (ctor, dtor, method, etc), it does not carry attributes
  // (__attribute__((...))), it does not track a class's base type, among other
  // details.
  static bool isCtorOrDtor() { return true; }
  static bool isMethod() { return true; }
  // NOTE(cir): This might not be necessary, since Clang queries Enums to find
  // their underlying integer type, which is already an int in CIR.
  static bool isEnum() { return true; }
  static bool isBuiltinType() { return true; }
  static bool isCXXRecord() { return true; }
  static bool recordBasesIterator() { return true; }
  static bool canPassInRegisters() { return true; }
  static bool alignmentAttribute() { return true; }
  static bool MSStructAttr() { return true; }
  static bool packedAttr() { return true; }
  static bool alignMac68kAttr() { return true; }
  static bool alignNaturalAttr() { return true; }
  static bool maxFieldAlignmentAttr() { return true; }
  static bool noUniqueAddressAttr() { return true; }

  // Abstraction to be created in CIR (not necessarily through interfaces).
  static bool fieldDeclAbs() { return true; } // Record field of any type.
  static bool qualTypeAbs() { return true; }  // Any qualified type.

  // Missing queries in CIR types.
  static bool getMaxAlignment() { return true; }
  static bool isPotentiallyOverlapping() { return true; }
  static bool isIntegralOrEnumerationType() { return true; }

  // Missing queries for field types in CIR.
  static bool isBitField() { return true; }
  static bool isUnnamedBitField() { return true; }

  // Missing queries for CIR CXX record types.
  static bool hasFlexibleArrayMember() { return true; }
  static bool isDynamicClass() { return true; }
  static bool mayInsertExtraPadding() { return true; }
  static bool isEmptyCXX11() { return true; }
  static bool isPODTR1() { return true; }

  // Missing queries for CIR array types.
  static bool isIncomplete() { return true; }

  // Some other possible source languages are not yet handled by CIR.
  static bool CUDA() { return true; }
  static bool Swift() { return true; }
  static bool ObjC() { return true; }

  // CIR does not yet hold any form of qualified types. This information is used
  // for ABI lowering and is stripped from the IR until only the canonical type
  // is left. We need to think about how to handle this.
  static bool qualifiedTypes() { return true; }

  // FunctionInfo objects are cached using a profile. This is not yet
  // implemented in CIR, and I'm not sure if it need to be.
  static bool fnInfoProfile() { return true; }

  // Clang uses the concept of "type classes" to distinguish between different
  // kinds of types, such as builtins, function, vectors, arrays, etc. This
  // distinction is not yet implemented in CIR.
  static bool typeClass() { return true; }

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
