#include "LoweringModule.h"
#include "CIRContext.h"
#include "LowerFunction.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace cir {

static CIRCXXABI *createCXXABI(LoweringModule &CGM) {
  switch (CGM.getCXXABIKind()) {
  case clang::TargetCXXABI::AppleARM64:
  case clang::TargetCXXABI::Fuchsia:
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::GenericARM:
  case clang::TargetCXXABI::iOS:
  case clang::TargetCXXABI::WatchOS:
  case clang::TargetCXXABI::GenericMIPS:
  case clang::TargetCXXABI::GenericItanium:
  case clang::TargetCXXABI::WebAssembly:
  case clang::TargetCXXABI::XL:
    return CreateItaniumCXXABI(CGM);
  case clang::TargetCXXABI::Microsoft:
    llvm_unreachable("Windows ABI NYI");
  }

  llvm_unreachable("invalid C++ ABI kind");
}

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LoweringModule &LM) {
  const clang::TargetInfo &Target = LM.getTarget();
  const llvm::Triple &Triple = Target.getTriple();

  switch (Triple.getArch()) {
  case llvm::Triple::x86_64: {
    // StringRef ABI = Target.getABI();
    // X86AVXABILevel AVXLevel = (ABI == "avx512" ? X86AVXABILevel::AVX512
    //                            : ABI == "avx"  ? X86AVXABILevel::AVX
    //                                            : X86AVXABILevel::None);

    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
      llvm_unreachable("Windows ABI NYI");
    default:
      return createX86_64TargetLoweringInfo(LM, X86AVXABILevel::None);
    }
  }
  default:
    llvm_unreachable("ABI NYI");
  }
}

LoweringModule::LoweringModule(CIRContext &C, ModuleOp &module, StringAttr DL,
                               const clang::TargetInfo &target)
    : context(C), module(module), Target(target), ABI(createCXXABI(*this)),
      types(*this, DL.getValue()) {}

const TargetLoweringInfo &LoweringModule::getTargetLoweringInfo() {
  if (!TheTargetCodeGenInfo)
    TheTargetCodeGenInfo = createTargetLoweringInfo(*this);
  return *TheTargetCodeGenInfo;
}

/// Returns the end of the function prologue.
///
/// The prologue is what is generated regardless of the function's body.
/// Arguments allocations for example. To identify this, this method uses a
/// naive approach of looking for the first store of the last argument.
static Block::iterator
setInsertionPointAtEndOfFunctionPrologue(FuncOp op, PatternRewriter &rewriter) {

  // Get the last argument.
  auto lastArg = op.getArguments().back();

  // Look for the first store of the last argument.
  for (auto &operand : lastArg.getUses())
    if (auto store = dyn_cast<StoreOp>(operand.getOwner()))
      rewriter.setInsertionPointAfter(store);

  llvm_unreachable("Could not find the end of the function prologue");
}

/// Rewrites an existing function to conform to the ABI (e.g. follow calling
/// conventions). This method tries to follow the original
/// CodeGenModule::EmitGlobalFunctionDefinition method as closely as possible.
/// However, they are inherently different.
void LoweringModule::rewriteGlobalFunctionDefinition(
    FuncOp op, LoweringModule &state, PatternRewriter &rewriter) {
  const LoweringFunctionInfo &FI =
      state.getTypes().arrangeGlobalDeclaration(op);
  FuncType Ty = state.getTypes().getFunctionType(FI);

  llvm::outs() << "Call Conv Lowering\n"
               << "\tfrom: " << op.getFunctionType() << "\n"
               << "\t  to: " << Ty << "\n";

  // NOTE(cir): Even if the old function type and the new function type are the
  // same, we still need to rewrite the function to conform to the ABI in most
  // cases. For example, `void (u32i)` will be lowered to `void (zeroext u32i)`.

  // FIXME(cir): The clone below might be flawed. For example, if a parameter
  // has an attribute but said parameter is coerced to multiple parameters, we
  // will not perform any for of mapping or attribute drop to account for
  // this. We need a proper procedure to rewrite a FuncOp and its properties
  // properly.
  FuncOp newFn = cast<FuncOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
  newFn.setType(Ty);

  LowerFunction(*this, rewriter, op).generateCode(op, newFn, FI);

  // Erase original ABI-agnostic function.
  rewriter.eraseOp(op);
}

} // namespace cir
} // namespace mlir
