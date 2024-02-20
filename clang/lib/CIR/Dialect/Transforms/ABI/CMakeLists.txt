add_clang_library(ABILowering
  LoweringTypes.cpp
  LoweringModule.cpp
  ABIInfo.cpp
  TargetLoweringInfo.cpp
  X86.cpp
  ABIInfoImpl.cpp
  TargetInfo.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  clangBasic

  MLIRIR
  MLIRPass

  MLIRCIR
  MLIRCIRInterfaces
)
