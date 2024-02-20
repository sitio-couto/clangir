#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Basic/TargetInfo.h"

class TestModule;

class TestTypes {
private:
  TestModule &LM;
  mlir::MLIRContext *ctx;

public:
  TestTypes(TestModule &LM);
  ~TestTypes() = default;
};

class TestModule {
private:
  mlir::MLIRContext *ctx;
  const clang::TargetInfo &Target;
  TestTypes types;

public:
  TestModule(mlir::ModuleOp &module, const clang::TargetInfo &target)
      : ctx(module.getContext()), Target(target), types(*this){};
  ~TestModule() = default;

  mlir::MLIRContext *getContext() { return ctx; }
};

TestTypes::TestTypes(TestModule &LM) : LM(LM), ctx(LM.getContext()){};
