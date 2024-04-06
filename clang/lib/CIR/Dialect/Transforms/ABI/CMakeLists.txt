add_clang_library(ABILowering
  LoweringTypes.cpp
  LoweringModule.cpp
  ABIInfo.cpp
  TargetLoweringInfo.cpp
  X86.cpp
  ABIInfoImpl.cpp
  TargetInfo.cpp
  ItaniumCXXABI.cpp
  CIRCXXABI.cpp
  CIRContext.cpp
  CIRRecordLayout.cpp
  RecordLayoutBuilder.cpp
  DataLayout.cpp
  LowerFunction.cpp

  DEPENDS

  LINK_LIBS PUBLIC

  MLIRIR
  MLIRPass

  MLIRCIR
  MLIRCIRInterfaces
)
