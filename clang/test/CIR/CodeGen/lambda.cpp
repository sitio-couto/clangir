// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-return-stack-address -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void fn() {
  auto a = [](){};
  a();
}

//      CHECK: !ty_anon2E0_ = !cir.struct<class "anon.0" {!u8i}>
//  CHECK-DAG: module

//      CHECK: cir.func lambda internal private @_ZZ2fnvENK3$_0clEv{{.*}}) extra

//      CHECK:   cir.func @_Z2fnv()
// CHECK-NEXT:     %0 = cir.alloca !ty_anon2E0_, !cir.ptr<!ty_anon2E0_>, ["a"]
//      CHECK:   cir.call @_ZZ2fnvENK3$_0clEv

void l0() {
  int i;
  auto a = [&](){ i = i + 1; };
  a();
}

// CHECK: cir.func lambda internal private @_ZZ2l0vENK3$_0clEv({{.*}}) extra

// CHECK: %0 = cir.alloca !cir.ptr<!ty_anon2E2_>, !cir.ptr<!cir.ptr<!ty_anon2E2_>>, ["this", init] {alignment = 8 : i64}
// CHECK: cir.store %arg0, %0 : !cir.ptr<!ty_anon2E2_>, !cir.ptr<!cir.ptr<!ty_anon2E2_>>
// CHECK: %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_anon2E2_>>, !cir.ptr<!ty_anon2E2_>
// CHECK: %2 = cir.get_member %1[0] {name = "i"} : !cir.ptr<!ty_anon2E2_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %3 = cir.load %2 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %4 = cir.load %3 : !cir.ptr<!s32i>, !s32i
// CHECK: %5 = cir.const #cir.int<1> : !s32i
// CHECK: %6 = cir.binop(add, %4, %5) nsw : !s32i
// CHECK: %7 = cir.get_member %1[0] {name = "i"} : !cir.ptr<!ty_anon2E2_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %8 = cir.load %7 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: cir.store %6, %8 : !s32i, !cir.ptr<!s32i>

// CHECK: cir.func @_Z2l0v()

auto g() {
  int i = 12;
  return [&] {
    i += 100;
    return i;
  };
}

// CHECK: cir.func @_Z1gv() -> !ty_anon2E3_
// CHECK: %0 = cir.alloca !ty_anon2E3_, !cir.ptr<!ty_anon2E3_>, ["__retval"] {alignment = 8 : i64}
// CHECK: %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK: %2 = cir.const #cir.int<12> : !s32i
// CHECK: cir.store %2, %1 : !s32i, !cir.ptr<!s32i>
// CHECK: %3 = cir.get_member %0[0] {name = "i"} : !cir.ptr<!ty_anon2E3_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK: cir.store %1, %3 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %4 = cir.load %0 : !cir.ptr<!ty_anon2E3_>, !ty_anon2E3_
// CHECK: cir.return %4 : !ty_anon2E3_

auto g2() {
  int i = 12;
  auto lam = [&] {
    i += 100;
    return i;
  };
  return lam;
}

// Should be same as above because of NRVO
// CHECK: cir.func @_Z2g2v() -> !ty_anon2E4_
// CHECK-NEXT: %0 = cir.alloca !ty_anon2E4_, !cir.ptr<!ty_anon2E4_>, ["__retval", init] {alignment = 8 : i64}
// CHECK-NEXT: %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK-NEXT: %2 = cir.const #cir.int<12> : !s32i
// CHECK-NEXT: cir.store %2, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %3 = cir.get_member %0[0] {name = "i"} : !cir.ptr<!ty_anon2E4_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.store %1, %3 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %4 = cir.load %0 : !cir.ptr<!ty_anon2E4_>, !ty_anon2E4_
// CHECK-NEXT: cir.return %4 : !ty_anon2E4_

int f() {
  return g2()();
}

//      CHECK: cir.func @_Z1fv() -> !s32i
// CHECK-NEXT:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %2 = cir.alloca !ty_anon2E4_, !cir.ptr<!ty_anon2E4_>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK-NEXT:     %3 = cir.call @_Z2g2v() : () -> !ty_anon2E4_
// CHECK-NEXT:     cir.store %3, %2 : !ty_anon2E4_, !cir.ptr<!ty_anon2E4_>
// CHECK-NEXT:     %4 = cir.call @_ZZ2g2vENK3$_0clEv(%2) : (!cir.ptr<!ty_anon2E4_>) -> !s32i
// CHECK-NEXT:     cir.store %4, %0 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.return %1 : !s32i
// CHECK-NEXT: }

int g3() {
  auto* fn = +[](int const& i) -> int { return i; };
  auto task = fn(3);
  return task;
}

// lambda operator()
// CHECK: cir.func lambda internal private @_ZZ2g3vENK3$_0clERKi{{.*}}!s32i extra

// lambda __invoke()
// CHECK:   cir.func internal private @_ZZ2g3vEN3$_08__invokeERKi

// lambda operator int (*)(int const&)()
// CHECK:   cir.func internal private @_ZZ2g3vENK3$_0cvPFiRKiEEv

// CHECK: cir.func @_Z2g3v() -> !s32i
// CHECK:     %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>>, ["fn", init] {alignment = 8 : i64}
// CHECK:     %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["task", init] {alignment = 4 : i64}

// 1. Use `operator int (*)(int const&)()` to retrieve the fnptr to `__invoke()`.
// CHECK:     %3 = cir.scope {
// CHECK:       %7 = cir.alloca !ty_anon2E5_, !cir.ptr<!ty_anon2E5_>, ["ref.tmp0"] {alignment = 1 : i64}
// CHECK:       %8 = cir.call @_ZZ2g3vENK3$_0cvPFiRKiEEv(%7) : (!cir.ptr<!ty_anon2E5_>) -> !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:       %9 = cir.unary(plus, %8) : !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:       cir.yield %9 : !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:     }

// 2. Load ptr to `__invoke()`.
// CHECK:     cir.store %3, %1 : !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>>
// CHECK:     %4 = cir.scope {
// CHECK:       %7 = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp1", init] {alignment = 4 : i64}
// CHECK:       %8 = cir.load %1 : !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>>, !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:       %9 = cir.const #cir.int<3> : !s32i
// CHECK:       cir.store %9, %7 : !s32i, !cir.ptr<!s32i>

// 3. Call `__invoke()`, which effectively executes `operator()`.
// CHECK:       %10 = cir.call %8(%7) : (!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!s32i>) -> !s32i
// CHECK:       cir.yield %10 : !s32i
// CHECK:     }

// CHECK:   }
