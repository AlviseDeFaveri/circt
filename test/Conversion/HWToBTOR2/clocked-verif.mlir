// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s

module {
  // CHECK: [[I32:[0-9]+]] sort bitvec 32
  // CHECK: [[A:[0-9]+]] input [[I32]] a
  // CHECK: [[B:[0-9]+]] input [[I32]] b
  hw.module @foo(in %clk : !seq.clock, in %a : i32, in %b : i32) {
    // Unnamed compregs (e.g. as created by other lowerings) get a fallback
    // name. The generic form is needed here since the custom parser always
    // attaches a name attribute.
    // CHECK: [[I1:[0-9]+]] sort bitvec 1
    // CHECK: [[REG:[0-9]+]] state [[I1]] compreg
    %reg = "seq.compreg"(%cond, %clk) {operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>} : (i1, !seq.clock) -> i1

    %0 = seq.from_clock %clk

    // CHECK: [[REM:[0-9]+]] urem [[I32]] [[A]] [[B]]
    %rem = comb.modu %a, %b : i32

    // comb.mods is a signed remainder (sign follows the dividend), i.e. srem
    // CHECK: [[SREM:[0-9]+]] srem [[I32]] [[A]] [[B]]
    %srem = comb.mods %a, %b : i32
    // CHECK: [[SUM:[0-9]+]] add [[I32]] [[REM]] [[SREM]]
    %sum = comb.add %rem, %srem : i32

    // CHECK: [[EQ:[0-9]+]] eq [[I1]] [[SUM]] [[A]]
    %cond = comb.icmp eq %sum, %a : i32

    // CHECK: [[C:[0-9]+]] constraint [[EQ]]
    verif.clocked_assume %cond, posedge %0 : i1

    // CHECK: [[NOT:[0-9]+]] not [[I1]] [[REG]]
    // CHECK: [[BAD:[0-9]+]] bad [[NOT]]{{$}}
    verif.clocked_assert %reg, posedge %0 : i1

    // Labels are emitted as the btor2 symbol of the bad/constraint line.
    // CHECK: constraint [[EQ]] Step0_MyAssume{{$}}
    verif.clocked_assume %cond, posedge %0 label "Step0_MyAssume" : i1
    // CHECK: [[NOT2:[0-9]+]] not [[I1]] [[REG]]
    // CHECK: bad [[NOT2]] Step1_MyAssert{{$}}
    verif.clocked_assert %reg, posedge %0 label "Step1_MyAssert" : i1
    // CHECK: [[NOT3:[0-9]+]] not [[I1]] [[REG]]
    // CHECK: bad [[NOT3]] has_spaces_in_label{{$}}
    verif.assert %reg label "has spaces in label" : i1

    // CHECK: next [[I1]] [[REG]] [[EQ]]
  }
}
