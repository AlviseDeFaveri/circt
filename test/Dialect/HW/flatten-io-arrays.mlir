// RUN: circt-opt --hw-flatten-io=flatten-arrays %s | FileCheck %s
// RUN: circt-opt --hw-flatten-io %s | FileCheck %s --check-prefix NOARR

// NOARR-LABEL: hw.module @arrayOfStruct(in %a : !hw.array<2xstruct<req: i12>>, out o : !hw.array<2xstruct<rsp: i1>>)

// CHECK-LABEL: hw.module @arrayOfStruct(in %a.0.req : i12, in %a.1.req : i12, out o.0.rsp : i1, out o.1.rsp : i1)
// CHECK:         %[[C:.+]] = hw.aggregate_constant
// CHECK-DAG:     %[[E0:.+]] = hw.array_get %[[C]][%false]
// CHECK-DAG:     %[[E1:.+]] = hw.array_get %[[C]][%true]
// CHECK-DAG:     %[[R0:.+]] = hw.struct_explode %[[E0]]
// CHECK-DAG:     %[[R1:.+]] = hw.struct_explode %[[E1]]
// CHECK:         hw.output %[[R0]], %[[R1]] : i1, i1
hw.module @arrayOfStruct(in %a : !hw.array<2xstruct<req: i12>>, out o : !hw.array<2xstruct<rsp: i1>>) {
  %c = hw.aggregate_constant [[false], [true]] : !hw.array<2xstruct<rsp: i1>>
  hw.output %c : !hw.array<2xstruct<rsp: i1>>
}

// CHECK-LABEL: hw.module @arrays(in %b.0 : i4, in %b.1 : i4, in %b.2 : i4, out p.0 : i4, out p.1 : i4, out p.2 : i4, out q : i4)
// CHECK:         %[[B:.+]] = hw.array_create %b.2, %b.1, %b.0 : i4
// CHECK:         hw.array_get %[[B]]
// CHECK:         %u.y.0, %u.y.1, %u.y.2 = hw.instance "u" @child(x.0: %b.0: i4, x.1: %b.1: i4, x.2: %b.2: i4) -> (y.0: i4, y.1: i4, y.2: i4)
// CHECK:         %[[Y:.+]] = hw.array_create %u.y.2, %u.y.1, %u.y.0 : i4
// CHECK-DAG:     %[[Y0:.+]] = hw.array_get %[[Y]][%c0_i2]
// CHECK-DAG:     %[[Y1:.+]] = hw.array_get %[[Y]][%c1_i2{{.*}}]
// CHECK-DAG:     %[[Y2:.+]] = hw.array_get %[[Y]][%c-2_i2]
// CHECK:         hw.output %[[Y0]], %[[Y1]], %[[Y2]], {{.+}} : i4, i4, i4, i4
hw.module @arrays(in %b : !hw.array<3xi4>, out p : !hw.array<3xi4>, out q : i4) {
  %i = hw.constant 1 : i2
  %e = hw.array_get %b[%i] : !hw.array<3xi4>, i2
  %r = hw.instance "u" @child(x: %b : !hw.array<3xi4>) -> (y: !hw.array<3xi4>)
  hw.output %r, %e : !hw.array<3xi4>, i4
}

// CHECK-LABEL: hw.module private @child(in %x.0 : i4, in %x.1 : i4, in %x.2 : i4, out y.0 : i4, out y.1 : i4, out y.2 : i4)
// CHECK-NEXT:    hw.output %x.0, %x.1, %x.2 : i4, i4, i4
hw.module private @child(in %x : !hw.array<3xi4>, out y : !hw.array<3xi4>) {
  hw.output %x : !hw.array<3xi4>
}

// CHECK-LABEL: hw.module @single(in %x.0 : i4, out y.0 : i4)
// CHECK-NEXT:    hw.output %x.0 : i4
hw.module @single(in %x : !hw.array<1xi4>, out y : !hw.array<1xi4>) {
  hw.output %x : !hw.array<1xi4>
}
