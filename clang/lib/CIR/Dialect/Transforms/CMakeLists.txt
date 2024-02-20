add_subdirectory(ABI)

add_clang_library(MLIRCIRTransforms
  LifetimeCheck.cpp
  LoweringPrepare.cpp
  LoweringPrepareItaniumCXXABI.cpp
  MergeCleanups.cpp
  DropAST.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  StdHelpers.cpp
  FlattenCFG.cpp
  CallConvLowering.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic
  ABILowering

  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRInterfaces
)
