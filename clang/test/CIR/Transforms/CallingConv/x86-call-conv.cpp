// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1
// RUN: FileCheck --input-file=%t.cir %s


/// Test trivial void case. ///

// CHECK: @_Z4Voidv()
void Void(void) {
// CHECK:   cir.call @_Z4Voidv() : () -> ()
  Void();
}


// Test call conv lowering for trivial zeroext cases.

// CHECK: cir.func @_Z5UCharh(%arg0: !u8i {cir.zeroext} loc({{.+}})) -> (!u8i {cir.zeroext})
unsigned char UChar(unsigned char c) {
  // CHECK: cir.call @_Z5UCharh(%2) : (!u8i) -> !u8i
  return UChar(c);
}
// CHECK: cir.func @_Z6UShortt(%arg0: !u16i {cir.zeroext} loc({{.+}})) -> (!u16i {cir.zeroext})
unsigned short UShort(unsigned short s) {
  // CHECK: cir.call @_Z6UShortt(%2) : (!u16i) -> !u16i
  return UShort(s);
}
// CHECK: cir.func @_Z4UIntj(%arg0: !u32i loc({{.+}})) -> !u32i
unsigned int UInt(unsigned int i) {
  // CHECK: cir.call @_Z4UIntj(%2) : (!u32i) -> !u32i
  return UInt(i);
}
// CHECK: cir.func @_Z5ULongm(%arg0: !u64i loc({{.+}})) -> !u64i
unsigned long ULong(unsigned long l) {
  // CHECK: cir.call @_Z5ULongm(%2) : (!u64i) -> !u64i
  return ULong(l);
}
// CHECK: cir.func @_Z9ULongLongy(%arg0: !u64i loc({{.+}})) -> !u64i
unsigned long long ULongLong(unsigned long long l) {
  // CHECK: cir.call @_Z9ULongLongy(%2) : (!u64i) -> !u64i
  return ULongLong(l);
}


/// Test call conv lowering for trivial signext cases. ///

// CHECK: cir.func @_Z4Chara(%arg0: !s8i {cir.signext} loc({{.+}})) -> (!s8i {cir.signext})
char Char(signed char c) {
  // CHECK: cir.call @_Z4Chara(%{{.+}}) : (!s8i) -> !s8i
  return Char(c);
}
// CHECK: cir.func @_Z5Shorts(%arg0: !s16i {cir.signext} loc({{.+}})) -> (!s16i {cir.signext})
short Short(short s) {
  // CHECK: cir.call @_Z5Shorts(%{{.+}}) : (!s16i) -> !s16i
  return Short(s);
}
// CHECK: cir.func @_Z3Inti(%arg0: !s32i loc({{.+}})) -> !s32i
int Int(int i) {
  // CHECK: cir.call @_Z3Inti(%{{.+}}) : (!s32i) -> !s32i
  return Int(i);
}
// CHECK: cir.func @_Z4Longl(%arg0: !s64i loc({{.+}})) -> !s64i
long Long(long l) {
  // CHECK: cir.call @_Z4Longl(%{{.+}}) : (!s64i) -> !s64i
  return Long(l);
}
// CHECK: cir.func @_Z8LongLongx(%arg0: !s64i loc({{.+}})) -> !s64i
long long LongLong(long long l) {
  // CHECK: cir.call @_Z8LongLongx(%{{.+}}) : (!s64i) -> !s64i
  return LongLong(l);
}


/// Test call conv lowering for floating point. ///

// CHECK: cir.func @_Z5Floatf(%arg0: !cir.float loc({{.+}})) -> !cir.float
float Float(float f) {
  // cir.call @_Z5Floatf(%{{.+}}) : (!cir.float) -> !cir.float
  return Float(f);
}
// CHECK: cir.func @_Z6Doubled(%arg0: !cir.double loc({{.+}})) -> !cir.double
double Double(double d) {
  // cir.call @_Z6Doubled(%{{.+}}) : (!cir.double) -> !cir.double
  return Double(d);
}


/// Test call conv lowering for trivial bitcast type coercion. ///

/// Cast argument to the expected type.
// CHECK: cir.func @_Z4Boolb(%arg0: !cir.int<u, 1> {cir.zeroext} loc({{.+}})) -> (!cir.int<u, 1> {cir.zeroext})
// CHECK: %[[#V0:]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>
// CHECK: %[[#V1:]] = cir.cast(bitcast, %arg0 : !cir.int<u, 1>), !cir.bool
// CHECK: cir.store %[[#V1]], %[[#V0]] : !cir.bool, !cir.ptr<!cir.bool>
bool Bool(bool a) {

  /// Cast argument and result of the function call to the expected types.
  // CHECK: %[[#V4:]] = cir.cast(bitcast, %{{.+}} : !cir.bool), !cir.int<u, 1>
  // CHECK: %[[#V5:]] = cir.call @_Z4Boolb(%[[#V4]]) : (!cir.int<u, 1>) -> !cir.int<u, 1>
  // CHECK: %{{.+}} = cir.cast(bitcast, %[[#V5]] : !cir.int<u, 1>), !cir.bool
  Bool(a);

  /// Cast return value to the expected type.
  // CHECK: %[[#V8:]] = cir.cast(bitcast, %{{.+}} : !cir.bool), !cir.int<u, 1>
  // CHECK: cir.return %[[#V8]] : !cir.int<u, 1>
  return a;
}


/// Test call conv lowering for pointer types. ///

// TODO(cir): Fix pointer call conv lowering.
// cir.func @Ptr(%arg0: !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {
//   cir.return %arg0 : !cir.ptr<!s32i>
// }


/// Test call conv lowering for struct type coercion scenarios. ///

struct S1 {
  int a, b;
};


/// Validate coerced argument and cast it to the expected type.

/// Cast arguments to the expected type.
// CHECK: cir.func @_Z2s12S1(%arg0: !u64i loc({{.+}})) -> !u64i
// CHECK: %[[#V0:]] = cir.alloca !ty_22S122, !cir.ptr<!ty_22S122>
// CHECK: %[[#V1:]] = cir.cast(bitcast, %arg0 : !u64i), !ty_22S122
// CHECK: cir.store %[[#V1]], %[[#V0]] : !ty_22S122, !cir.ptr<!ty_22S122>
S1 s1(S1 arg) {

  /// Cast argument and result of the function call to the expected types.
  // CHECK: %[[#V9:]] = cir.cast(bitcast, %{{.+}} : !ty_22S122), !u64i
  // CHECK: %[[#V10:]] = cir.call @_Z2s12S1(%[[#V9]]) : (!u64i) -> !u64i
  // CHECK: %[[#V11:]] = cir.cast(bitcast, %[[#V10]] : !u64i), !ty_22S122
  s1({1, 2});

  // CHECK: %[[#V12:]] = cir.load %{{.+}} : !cir.ptr<!ty_22S122>, !ty_22S122
  // CHECK: %[[#V13:]] = cir.cast(bitcast, %[[#V12]] : !ty_22S122), !u64i
  // CHECK: cir.return %[[#V13]] : !u64i
  return {1, 2};
}
