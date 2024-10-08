; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -mtriple=hexagon-- -passes=atomic-expand %s | FileCheck %s

define float @test_atomicrmw_fadd_f32(ptr %ptr, float %value) {
; CHECK-LABEL: @test_atomicrmw_fadd_f32(
; CHECK-NEXT:    br label [[ATOMICRMW_START:%.*]]
; CHECK:       atomicrmw.start:
; CHECK-NEXT:    [[LARX:%.*]] = call i32 @llvm.hexagon.L2.loadw.locked(ptr [[PTR:%.*]])
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32 [[LARX]] to float
; CHECK-NEXT:    [[NEW:%.*]] = fadd float [[TMP1]], [[VALUE:%.*]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast float [[NEW]] to i32
; CHECK-NEXT:    [[STCX:%.*]] = call i32 @llvm.hexagon.S2.storew.locked(ptr [[PTR]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i32 [[STCX]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = zext i1 [[TMP3]] to i32
; CHECK-NEXT:    br i1 [[TMP3]], label [[ATOMICRMW_START]], label [[ATOMICRMW_END:%.*]]
; CHECK:       atomicrmw.end:
; CHECK-NEXT:    ret float [[TMP1]]
;
  %res = atomicrmw fadd ptr %ptr, float %value seq_cst
  ret float %res
}

define float @test_atomicrmw_fsub_f32(ptr %ptr, float %value) {
; CHECK-LABEL: @test_atomicrmw_fsub_f32(
; CHECK-NEXT:    br label [[ATOMICRMW_START:%.*]]
; CHECK:       atomicrmw.start:
; CHECK-NEXT:    [[LARX:%.*]] = call i32 @llvm.hexagon.L2.loadw.locked(ptr [[PTR:%.*]])
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32 [[LARX]] to float
; CHECK-NEXT:    [[NEW:%.*]] = fsub float [[TMP1]], [[VALUE:%.*]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast float [[NEW]] to i32
; CHECK-NEXT:    [[STCX:%.*]] = call i32 @llvm.hexagon.S2.storew.locked(ptr [[PTR]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i32 [[STCX]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = zext i1 [[TMP3]] to i32
; CHECK-NEXT:    br i1 [[TMP3]], label [[ATOMICRMW_START]], label [[ATOMICRMW_END:%.*]]
; CHECK:       atomicrmw.end:
; CHECK-NEXT:    ret float [[TMP1]]
;
  %res = atomicrmw fsub ptr %ptr, float %value seq_cst
  ret float %res
}

