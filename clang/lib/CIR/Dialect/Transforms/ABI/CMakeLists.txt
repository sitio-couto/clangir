add_clang_library(ABILowering
  LoweringTypes.cpp
  LoweringModule.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  clangBasic

  MLIRIR
  MLIRPass

  MLIRCIR
  MLIRCIRInterfaces
)
