// RUN: circt-opt --llhd-remove-control-flow %s | FileCheck %s

// CHECK-LABEL: @Basic
hw.module @Basic(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: [[TMP0:%.+]] = comb.icmp eq %a, %b
    %0 = comb.icmp eq %a, %b : i42
    // CHECK-NEXT: [[TMP1:%.+]] = comb.add %a, %b
    %1 = comb.add %a, %b : i42
    // CHECK-NOT: cf.br
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: [[TMP2:%.+]] = comb.sub %a, %b
    %2 = comb.sub %a, %b : i42
    // CHECK-NOT: cf.cond_br
    cf.cond_br %0, ^bb2(%1 : i42), ^bb3(%2 : i42)
  ^bb2(%3: i42):
    // CHECK-NEXT: [[TMP4:%.+]] = comb.mul [[TMP1]], [[TMP2]]
    %4 = comb.mul %3, %2 : i42
    // CHECK-NEXT: [[TMP5:%.+]] = comb.icmp eq [[TMP4]], %b
    %5 = comb.icmp eq %4, %b : i42
    // CHECK-NOT: cf.cond_br
    cf.cond_br %5, ^bb3(%4 : i42), ^bb4(%3 : i42)
  ^bb3(%6: i42):
    // CHECK-NEXT: [[TMP6:%.+]] = comb.mux [[TMP0]], [[TMP4]], [[TMP2]]
    // CHECK-NEXT: [[TMP7:%.+]] = comb.xor [[TMP1]], [[TMP6]]
    %7 = comb.xor %1, %6 : i42
    // CHECK-NOT: cf.br
    cf.br ^bb4(%7 : i42)
  ^bb4(%8: i42):
    // CHECK-NEXT: [[TMP8A:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP8B:%.+]] = comb.xor [[TMP5]], [[TMP8A]]
    // CHECK-NEXT: [[TMP8C:%.+]] = comb.and [[TMP0]], [[TMP8B]]
    // CHECK-NEXT: [[TMP8:%.+]] = comb.mux [[TMP8C]], [[TMP1]], [[TMP7]]
    // CHECK-NEXT: llhd.yield [[TMP8]]
    llhd.yield %8 : i42
  }
}

// CHECK-LABEL: @SkipWhenSideEffectsPresent
hw.module @SkipWhenSideEffectsPresent() {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational {
    // CHECK: cf.br
    cf.br ^bb1
  ^bb1:
    func.call @someFunc() : () -> ()
    llhd.yield
  }
}

// CHECK-LABEL: @SkipWhenLoopsPresent
hw.module @SkipWhenLoopsPresent(in %a: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational {
    // CHECK: cf.br
    cf.br ^bb1
  ^bb1:
    // CHECK: cf.cond_br
    cf.cond_br %a, ^bb1, ^bb2
  ^bb2:
    llhd.yield
  }
}

// CHECK-LABEL: @IgnoreValuesComingFromUnreachableBlock
hw.module @IgnoreValuesComingFromUnreachableBlock(in %a: i42, in %b: i42) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: llhd.yield %a
    cf.br ^bb1(%a : i42)
  ^bb1(%0: i42):
    llhd.yield %0 : i42
  ^bb2:
    cf.br ^bb1(%b : i42)
  }
}

// CHECK-LABEL: @MultipleYields
hw.module @MultipleYields(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: [[TMP:%.+]] = comb.mux %c, %a, %b
    // CHECK-NEXT: llhd.yield [[TMP]]
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    llhd.yield %a : i42
  ^bb2:
    llhd.yield %b : i42
  }
}

// CHECK-LABEL: @HandleCondBranchToSameBlock
hw.module @HandleCondBranchToSameBlock(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: [[TMP:%.+]] = comb.mux %c, %a, %b
    // CHECK-NEXT: llhd.yield [[TMP]]
    cf.cond_br %c, ^bb1(%a : i42), ^bb1(%b : i42)
  ^bb1(%0: i42):
    llhd.yield %0 : i42
  }
}

// Merged values must not depend on branch decisions that do not affect them,
// since that may create spurious combinational loops. Here `%stall` must only
// affect `%ns`. `%run` is the same on all paths through `^bb1`, and `%flush`
// takes its most common value on all of them, so neither should depend on
// `%stall`.
// CHECK-LABEL: @NoSpuriousDependencies
hw.module @NoSpuriousDependencies(in %state: i1, in %mode: i1, in %stall: i1, in %a: i42, in %b: i42) {
  // CHECK-NEXT: llhd.combinational
  // CHECK-NOT: %stall
  // CHECK: [[NOTSTATE:%.+]] = comb.xor %state, {{%.+}} : i1
  // CHECK-NEXT: [[FLUSH:%.+]] = comb.and %mode, [[NOTSTATE]] : i1
  // CHECK-NOT: %stall
  // CHECK: [[NOTSTALL:%.+]] = comb.xor %stall, {{%.+}} : i1
  // CHECK-NEXT: [[COND:%.+]] = comb.and %state, [[NOTSTALL]] : i1
  // CHECK-NEXT: [[NS:%.+]] = comb.mux [[COND]], %a, %b : i42
  // CHECK-NEXT: llhd.yield [[FLUSH]], %state, [[NS]] : i1, i1, i42
  llhd.combinational -> i1, i1, i42 {
    %false = hw.constant false
    %true = hw.constant true
    cf.cond_br %state, ^bb1, ^bb3
  ^bb1:
    cf.cond_br %stall, ^bb2, ^bb6(%false, %true, %a : i1, i1, i42)
  ^bb2:
    cf.br ^bb6(%false, %true, %b : i1, i1, i42)
  ^bb3:
    cf.cond_br %mode, ^bb4, ^bb5
  ^bb4:
    cf.br ^bb6(%true, %false, %b : i1, i1, i42)
  ^bb5:
    cf.br ^bb6(%false, %false, %b : i1, i1, i42)
  ^bb6(%flush: i1, %run: i1, %ns: i42):
    llhd.yield %flush, %run, %ns : i1, i1, i42
  }
}

func.func private @someFunc()
