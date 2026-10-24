// RUN: circt-opt --seq-compreg-to-firreg %s | FileCheck %s

// CHECK-LABEL: hw.module @Basic
hw.module @Basic(in %clk: !seq.clock, in %rst: i1, in %a: i8, out o0: i8, out o1: i8) {
  // CHECK-NOT: seq.compreg
  // CHECK: %r0 = seq.firreg %a clock %clk : i8
  %r0 = seq.compreg %a, %clk : i8
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i8
  // CHECK: %r1 = seq.firreg %a clock %clk reset sync %rst, [[ZERO]] : i8
  %c0_i8 = hw.constant 0 : i8
  %r1 = seq.compreg %a, %clk reset %rst, %c0_i8 : i8
  hw.output %r0, %r1 : i8, i8
}

// Constant initial values become presets, and the initial op is removed once
// no register uses it anymore.
// CHECK-LABEL: hw.module @Initial
hw.module @Initial(in %clk: !seq.clock, in %rst: i1, in %a: i8, out o0: i8, out o1: i8) {
  // CHECK-NOT: seq.initial
  // CHECK: %r0 = seq.firreg %a clock %clk preset 5 : i8
  // CHECK: %r1 = seq.firreg %a clock %clk reset sync %rst, %{{.+}} preset 5 : i8
  %init = seq.initial () {
    %c5_i8 = hw.constant 5 : i8
    seq.yield %c5_i8 : i8
  } : () -> !seq.immutable<i8>
  %c0_i8 = hw.constant 0 : i8
  %r0 = seq.compreg %a, %clk initial %init : i8
  %r1 = seq.compreg %a, %clk reset %rst, %c0_i8 initial %init : i8
  hw.output %r0, %r1 : i8, i8
}

// Initial values that are not constants cannot be presets.
// CHECK-LABEL: hw.module @NonConstantInitial
hw.module @NonConstantInitial(in %clk: !seq.clock, in %a: i8, out o: i8) {
  // CHECK: seq.initial
  // CHECK: %r = seq.compreg %a, %clk initial %{{.+}} : i8
  %init = seq.initial () {
    %0 = sim.func.dpi.call @random() : () -> i8
    seq.yield %0 : i8
  } : () -> !seq.immutable<i8>
  %r = seq.compreg %a, %clk initial %init : i8
  hw.output %r : i8
}
sim.func.dpi private @random(out result : i8)
