// RUN: circt-opt %s --convert-hw-to-btor2 -o /dev/null | FileCheck %s
// RUN: circt-opt %s --convert-hw-to-btor2="assume-init-reset" -o /dev/null | FileCheck %s --check-prefix=RESET

// A firreg preset is the register's initial value and must be emitted as an
// init statement (constant before the state, init after it).

// CHECK: [[I32:[0-9]+]] sort bitvec 32
// CHECK: [[I1:[0-9]+]] sort bitvec 1
// CHECK: [[C0:[0-9]+]] constd [[I1]] 0
// CHECK: [[RST:[0-9]+]] state [[I1]] rst_ni
// CHECK: init [[I1]] [[RST]] [[C0]]
// CHECK: [[C5:[0-9]+]] constd [[I32]] 5
// CHECK: [[P:[0-9]+]] state [[I32]] preset_reg
// CHECK: init [[I32]] [[P]] [[C5]]
// CHECK: [[C4294967295:[0-9]+]] constd [[I32]] 4294967295
// CHECK: [[N:[0-9]+]] state [[I32]] neg_preset_reg
// CHECK: init [[I32]] [[N]] [[C4294967295]]
// Without assume-init-reset, the preset is used even if there is a reset.
// CHECK: [[R:[0-9]+]] state [[I32]] reset_reg
// CHECK: init [[I32]] [[R]] [[C5]]
// A register without preset starts with an arbitrary value.
// CHECK: [[U:[0-9]+]] state [[I32]] plain_reg
// CHECK-NOT: init [[I32]] [[U]]

// With assume-init-reset, a constant reset value takes priority over the
// preset.
// RESET: [[I32:[0-9]+]] sort bitvec 32
// RESET: [[C2:[0-9]+]] constd [[I32]] 2
// RESET: [[R:[0-9]+]] state [[I32]] reset_reg
// RESET: init [[I32]] [[R]] [[C2]]
// RESET: [[U:[0-9]+]] state [[I32]] plain_reg
// RESET-NOT: init [[I32]] [[U]]

module {
  hw.module @top(in %clk : !seq.clock, in %a : i32, in %rst : i1) {
    %true = hw.constant true
    %c2_i32 = hw.constant 2 : i32
    %rst_ni = seq.firreg %true clock %clk preset 0 : i1
    %preset_reg = seq.firreg %a clock %clk preset 5 : i32
    %neg_preset_reg = seq.firreg %a clock %clk preset 4294967295 : i32
    %reset_reg = seq.firreg %a clock %clk reset sync %rst, %c2_i32 preset 5 : i32
    %plain_reg = seq.firreg %a clock %clk : i32
  }
}
