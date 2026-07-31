// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialized value.
// UNSUPPORTED: valgrind

// Slang models a labeled statement as a sequential block wrapped around the
// statement itself. A labeled concurrent assertion must still be emitted
// directly into the module body, the same way an unlabeled one is, instead of
// being buried in an `always` procedure that no later pass can dissolve.

// CHECK-LABEL: moore.module @Unlabeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert
// CHECK-NOT: label
module Unlabeled(input logic clk, input logic a, input logic b);
  assert property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @Labeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert {{.*}} label "my_assert"
module Labeled(input logic clk, input logic a, input logic b);
  my_assert: assert property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @LabeledAssume
// CHECK-NOT: moore.procedure
// CHECK: verif.assume {{.*}} label "my_assume"
module LabeledAssume(input logic clk, input logic a, input logic b);
  my_assume: assume property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @LabeledCover
// CHECK-NOT: moore.procedure
// CHECK: verif.cover {{.*}} label "my_cover"
module LabeledCover(input logic clk, input logic a, input logic b);
  my_cover: cover property (@(posedge clk) a && b);
endmodule

// Several labeled assertions in one module all land in the module body.
// CHECK-LABEL: moore.module @ManyLabeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert {{.*}} label "first"
// CHECK: verif.assert {{.*}} label "second"
// CHECK: verif.assert {{.*}} label "third"
module ManyLabeled(input logic clk, input logic a, input logic b);
  first:  assert property (@(posedge clk) a |-> b);
  second: assert property (@(posedge clk) b |-> a);
  third:  assert property (@(posedge clk) a);
endmodule

// A procedure that is not a concurrent assertion keeps its procedure.
// CHECK-LABEL: moore.module @OrdinaryProcedure
// CHECK: moore.procedure always_comb
module OrdinaryProcedure(input logic a, output logic b);
  always_comb begin
    b = a;
  end
endmodule

// Slang wraps a module-level deferred immediate assertion in the same synthetic
// `always` procedure. Check that this isn't stripped.
// CHECK-LABEL: moore.module @DeferredImmediate
// CHECK: moore.procedure always {
// CHECK: moore.assert final
// CHECK: moore.procedure always {
// CHECK: moore.assume final
// CHECK: moore.procedure always {
// CHECK: moore.cover final
module DeferredImmediate(input logic a, input logic b);
  assert final (a);
  assume final (b);
  cover final (a && b);
endmodule

// Labels carry the hierarchical path of the assertion within its module,
// including generate blocks and named blocks. Generate block array entries are
// named the same way as the variables declared inside of them.
// CHECK-LABEL: moore.module @HierarchicalLabels
module HierarchicalLabels(input logic clk, input logic a, input logic b);
  // CHECK: verif.assert {{.*}} label "gen_arr_0.arr_assert"
  // CHECK: verif.assert {{.*}} label "gen_arr_1.arr_assert"
  for (genvar i = 0; i < 2; i++) begin : gen_arr
    arr_assert: assert property (@(posedge clk) a);
  end
  // CHECK: verif.assert {{.*}} label "gen_if.if_assert"
  if (1) begin : gen_if
    if_assert: assert property (@(posedge clk) b);
  end
  // CHECK: moore.procedure always
  // CHECK: verif.assert {{.*}} label "proc_assert"
  always @(posedge clk) proc_assert: assert property (a);
  // CHECK: moore.procedure always_comb
  // CHECK: moore.assert immediate {{.*}} label "blk.imm_assert"
  // CHECK: moore.assume immediate {{.*}} label "blk.imm_assume"
  // CHECK: moore.cover immediate {{.*}} label "blk.imm_cover"
  always_comb begin : blk
    imm_assert: assert (a | b);
    imm_assume: assume (a);
    imm_cover: cover (b);
  end
endmodule

// A label inside the action block of an unlabeled assertion only applies to
// the nested assertion.
// CHECK-LABEL: moore.module @NestedLabel
module NestedLabel(input logic a, input logic b);
  // CHECK: moore.procedure always_comb
  // CHECK-NOT: label
  // CHECK: moore.assert immediate {{.*}} label "inner"
  always_comb
    assert (a) else inner: assert (b);
endmodule

// Interface instances are expanded into the enclosing module, so labels inside
// them are prefixed with the instance name like the interface's signals.
interface LabeledIntf(input logic clk, input logic a);
  intf_assume: assume property (@(posedge clk) a);
endinterface

// CHECK-LABEL: moore.module @InterfaceLabels
module InterfaceLabels(input logic clk, input logic a, input logic b);
  // CHECK: verif.assume {{.*}} label "intf0_intf_assume"
  // CHECK: verif.assume {{.*}} label "intf1_intf_assume"
  LabeledIntf intf0(.clk, .a);
  LabeledIntf intf1(.clk, .a(b));
endmodule
