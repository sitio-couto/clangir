//===--- MissingFeatures.cpp - Markers for missing C/C++ features in CIR --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static methods declared here are spread throughout the CIR lowering codebase
// to track missing features in CIR that should eventually be implemented. Add
// new methods as needed.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
namespace cir {

struct MissingFeature {

  //==-- Missing languages -------------------------------------------------==//

  static bool CUDA() { return true; }
  static bool swift() { return true; }
  static bool OpenMP() { return true; }

  //==-- Missing AST queries -----------------------------------------------==//

  static bool funcDeclIsCXXConstructorDecl() { return true; }
  static bool funcDeclIsCXXDestructorDecl() { return true; }
  static bool funcDeclIsCXXMethodDecl() { return true; }
  static bool funcDeclIsInlineBuiltinDeclaration() { return true; }
  static bool funcDeclIsReplaceableGlobalAllocationFunction() { return true; }
  static bool qualTypeIsReferenceType() { return true; }
  static bool recordDeclCanPassInRegisters() { return true; }

  //==-- Missing types -----------------------------------------------------==//

  static bool vectorType() { return true; }

  //==-- Other missing features --------------------------------------------==//

  // Calls with a static chain pointer argument may be optimized (p.e. freeing
  // up argument registers), but we do not yet track such cases.
  static bool chainCall() { return true; }

  // Parameters may have additional attributes (e.g. [[noescape]],
  // [[callback(0)]]) that affect the compiler. This is NYI in CIR.
  static bool extParamInfo() { return true; }

  // Inalloca parameter attributes are mostly used for Windows x86_32 ABI. We
  // do not yet support this yet.
  static bool inallocaArgs() { return true; }

  // LangOpts may affect lowering, but we do not carry this information into CIR
  // just yet. Right now, it only instantiates the default lang options.
  static bool langOpts() { return true; }

  // Several type qualifiers are not yet supported in CIR, but important when
  // evaluating ABI-specific lowering.
  static bool qualifiedTypes() { return true; }

  // ABI-lowering has special handling for regcall calling convention (tries to
  // pass everyargument in regs). We don't support it just yet.
  static bool regCall() { return true; }

  // Some ABIs (e.g. x86) require special handling for returning large structs
  // by value. The sret argument parameter aids in this, but it is current NYI.
  static bool sretArgs() { return true; }

  static bool noReturn() { return true; }
  static bool csmeCall() { return true; }

  // Despite carrying some information about variadics, we are currently
  // ignoring this to focus only on the code necessary to lower non-variadics.
  static bool variadicFunctions() { return true; }
};

} // namespace cir
} // namespace mlir
