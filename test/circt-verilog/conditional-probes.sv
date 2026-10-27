// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang

// Hierarchical references reach the parent module as `!llhd.ref` ports, which
// Mem2Reg cannot promote. Reading such a reference under a condition leaves a
// probe in a block that is not directly entered from an `llhd.wait`, where
// HoistSignals does not hoist it. Deseq must still turn the `always_ff` into a
// register, and the `always_comb` must still collapse into plain dataflow.

module Child(input logic [7:0] x);
  logic [7:0] internal;
  assign internal = x + 8'd1;
endmodule

// CHECK-LABEL: hw.module @ConditionalProbes(
module ConditionalProbes(
  input logic clock,
  input logic en,
  input logic [7:0] x,
  output logic [7:0] q,
  output logic [7:0] y
);
  // CHECK-NOT: llhd.process
  // CHECK-NOT: llhd.combinational
  // CHECK: hw.instance "child" @Child
  // CHECK-NOT: llhd.process
  // CHECK-NOT: llhd.combinational
  // CHECK-DAG: [[PRB:%.+]] = llhd.prb %child.internal : i8
  // CHECK-DAG: [[Y:%.+]] = comb.mux %en, [[PRB]], %c0_i8 : i8
  // CHECK-DAG: [[NEXT:%.+]] = comb.mux bin %en, [[PRB]], [[Q:%.+]] : i8
  // CHECK-DAG: [[Q]] = seq.firreg [[NEXT]] clock %clock : i8
  // CHECK: hw.output [[Q]], [[Y]]
  Child child(.x(x));

  always_ff @(posedge clock)
    if (en)
      q <= child.internal;

  always_comb begin
    y = 0;
    if (en)
      y = child.internal;
  end
endmodule
