// RUN: circt-opt --llhd-sig2reg -cse %s | FileCheck %s

func.func @getTime() -> !llhd.time {
  %time = llhd.constant_time <1ns, 0d, 0e>
  return %time : !llhd.time
}

hw.module @basic(in %init : i32, in %cond : i1, in %in0 : i32, in %in1 : i32, out prb1 : i32, out prb2 : i32, out prb3 : i32, out prb4 : i32, out prb5 : i32, out prb6 : i32, out prb7 : i32) {
  %opaque_time = func.call @getTime() : () -> !llhd.time
  %epsilon = llhd.constant_time <0ns, 0d, 1e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  // Promoted without delay op
  %sig1 = llhd.sig %init : i32
  // Promoted with delay op
  %sig2 = llhd.sig %init : i32
  // Not promoted because time is opaque, can support if the delay op takes a
  // time value instead of attribute
  %sig3 = llhd.sig %init : i32
  // Promoted to %init because no drive present
  %sig4 = llhd.sig %init : i32
  // Not promoted because of drive condition
  %sig5 = llhd.sig %init : i32
  // Not promoted because a user is in a nested region
  %sig6 = llhd.sig %init : i32
  // Not promoted because of multiple drivers
  %sig7 = llhd.sig %init : i32

  llhd.drv %sig1, %in0 after %epsilon : i32
  // CHECK: [[DELAY:%.+]] = llhd.delay %in0 by <0ns, 1d, 0e> : i32
  llhd.drv %sig2, %in0 after %delta : i32
  llhd.drv %sig3, %in0 after %opaque_time : i32

  llhd.drv %sig5, %in0 after %epsilon if %cond : i32

  scf.if %cond {
    llhd.drv %sig6, %in0 after %epsilon : i32
  }
  
  llhd.drv %sig7, %in0 after %epsilon : i32
  llhd.drv %sig7, %in1 after %delta : i32

  %prb1 = llhd.prb %sig1 : i32
  %prb2 = llhd.prb %sig2 : i32
  // CHECK: [[PRB3:%.+]] = llhd.prb %sig3
  %prb3 = llhd.prb %sig3 : i32
  %prb4 = llhd.prb %sig4 : i32
  // CHECK: [[PRB5:%.+]] = llhd.prb %sig5
  %prb5 = llhd.prb %sig5 : i32
  // CHECK: [[PRB6:%.+]] = llhd.prb %sig6
  %prb6 = llhd.prb %sig6 : i32
  // CHECK: [[PRB7:%.+]] = llhd.prb %sig7
  %prb7 = llhd.prb %sig7 : i32

  // CHECK: hw.output %in0, [[DELAY]], [[PRB3]], %init, [[PRB5]], [[PRB6]], [[PRB7]] :
  hw.output %prb1, %prb2, %prb3, %prb4, %prb5, %prb6, %prb7 : i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: hw.module @aliasStatic
hw.module @aliasStatic(in %init : i4, in %in0 : i1, in %in1 : i1, out out: i4) {
  // CHECK-NEXT: [[C_2_I4:%.+]] = hw.constant -2 : i4
  // CHECK-NEXT: [[V0:%.+]] = comb.and %init, [[C_2_I4]] : i4
  // CHECK-NEXT: [[C0_I3:%.+]] = hw.constant 0 : i3
  // CHECK-NEXT: [[V1:%.+]] = comb.concat [[C0_I3]], %in1 : i3, i1
  // CHECK-NEXT: [[V2:%.+]] = comb.or [[V1]], [[V0]] : i4
  // CHECK-NEXT: [[C1_I4:%.+]] = hw.constant 1 : i4
  // CHECK-NEXT: [[C_3_I4:%.+]] = hw.constant -3 : i4
  // CHECK-NEXT: [[V3:%.+]] = comb.and [[V2]], [[C_3_I4]] : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.concat [[C0_I3]], %in0 : i3, i1
  // CHECK-NEXT: [[V5:%.+]] = comb.shl [[V4]], [[C1_I4]] : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.or [[V5]], [[V3]] : i4
  // CHECK-NEXT: [[C2_I4:%.+]] = hw.constant 2 : i4
  // CHECK-NEXT: [[C_5_I4:%.+]] = hw.constant -5 : i4
  // CHECK-NEXT: [[V7:%.+]] = comb.and [[V6]], [[C_5_I4]] : i4
  // CHECK-NEXT: [[V8:%.+]] = comb.shl [[V1]], [[C2_I4]] : i4
  // CHECK-NEXT: [[V9:%.+]] = comb.or [[V8]], [[V7]] : i4
  // CHECK-NEXT: hw.output [[V9]] : i4

  %0 = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %c1_c2 = hw.constant 1 : i2
  %c0_c2 = hw.constant 0 : i2
  %false = hw.constant false
  %out = llhd.sig %init : i4
  %3 = llhd.sig.extract %out from %c1_c2 : <i4> -> <i2>
  %6 = llhd.sig.extract %3 from %false : <i2> -> <i1>
  %7 = llhd.sig.extract %3 from %true : <i2> -> <i1>
  llhd.drv %6, %in0 after %0 : i1
  llhd.drv %7, %in1 after %0 : i1
  %4 = llhd.sig.extract %out from %c0_c2 : <i4> -> <i1>
  llhd.drv %4, %in1 after %0 : i1
  %5 = llhd.prb %out : i4
  hw.output %5 : i4
}

// CHECK-LABEL: hw.module @aliasDynamicSuccess
hw.module @aliasDynamicSuccess(in %init : i8, in %in0 : i1, in %in1 : i1, in %idx0 : i2, in %idx1 : i1, out out: i8) {
  // CHECK-NEXT: [[C1_I8:%.+]] = hw.constant 1 : i8
  // CHECK-NEXT: [[C0_I8:%.+]] = hw.constant 0 : i8
  // CHECK-NEXT: [[C0_I6:%.+]] = hw.constant 0 : i6
  // CHECK-NEXT: [[V0:%.+]] = comb.concat [[C0_I6]], %idx0 : i6, i2
  // CHECK-NEXT: [[V1:%.+]] = comb.add [[V0]], [[C0_I8]] : i8
  // CHECK-NEXT: [[C0_I7:%.+]] = hw.constant 0 : i7
  // CHECK-NEXT: [[V2:%.+]] = comb.concat [[C0_I7]], %idx1 : i7, i1
  // CHECK-NEXT: [[V3:%.+]] = comb.add [[V1]], [[V2]] : i8
  // CHECK-NEXT: [[V4:%.+]] = comb.shl [[C1_I8]], [[V3]] : i8
  // CHECK-NEXT: [[C_1_I8:%.+]] = hw.constant -1 : i8
  // CHECK-NEXT: [[V5:%.+]] = comb.xor [[V4]], [[C_1_I8]] : i8
  // CHECK-NEXT: [[V6:%.+]] = comb.and %init, [[V5]] : i8
  // CHECK-NEXT: [[V7:%.+]] = comb.concat [[C0_I7]], %in0 : i7, i1
  // CHECK-NEXT: [[V8:%.+]] = comb.shl [[V7]], [[V3]] : i8
  // CHECK-NEXT: [[V9:%.+]] = comb.or [[V8]], [[V6]] : i8
  // CHECK-NEXT: [[C4_I8:%.+]] = hw.constant 4 : i8
  // CHECK-NEXT: [[V10:%.+]] = comb.add [[V0]], [[C4_I8]] : i8
  // CHECK-NEXT: [[V11:%.+]] = comb.shl [[C1_I8]], [[V10]] : i8
  // CHECK-NEXT: [[V12:%.+]] = comb.xor [[V11]], [[C_1_I8]] : i8
  // CHECK-NEXT: [[V13:%.+]] = comb.and [[V9]], [[V12]] : i8
  // CHECK-NEXT: [[V14:%.+]] = comb.concat [[C0_I7]], %in1 : i7, i1
  // CHECK-NEXT: [[V15:%.+]] = comb.shl [[V14]], [[V10]] : i8
  // CHECK-NEXT: [[V16:%.+]] = comb.or [[V15]], [[V13]] : i8
  // CHECK-NEXT: hw.output [[V16]] : i8

  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c4_c3 = hw.constant 4 : i3
  %c0_c3 = hw.constant 0 : i3
  %out = llhd.sig %init : i8
  %3 = llhd.sig.extract %out from %c0_c3 : <i8> -> <i4>
  %4 = llhd.sig.extract %out from %c4_c3 : <i8> -> <i4>
  %5 = llhd.sig.extract %3 from %idx0 : <i4> -> <i2>
  %6 = llhd.sig.extract %5 from %idx1 : <i2> -> <i1>
  %7 = llhd.sig.extract %4 from %idx0 : <i4> -> <i1>
  llhd.drv %6, %in0 after %0 : i1
  llhd.drv %7, %in1 after %0 : i1
  %8 = llhd.prb %out : i8
  hw.output %8 : i8
}

// CHECK-LABEL: hw.module @aliasDynamicFailure
hw.module @aliasDynamicFailure(in %init : i4, in %in0 : i1, in %in1 : i1, in %idx0 : i1, in %idx1 : i1, out out: i4) {
  // CHECK-NEXT: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: [[C_2_I2:%.+]] = hw.constant -2 : i2
  // CHECK-NEXT: [[C0_I2:%.+]] = hw.constant 0 : i2
  // CHECK-NEXT: [[OUT:%.+]] = llhd.sig %init : i4
  // CHECK-NEXT: [[V1:%.+]] = llhd.sig.extract [[OUT]] from [[C0_I2]] : <i4> -> <i2>
  // CHECK-NEXT: [[V2:%.+]] = llhd.sig.extract [[OUT]] from [[C_2_I2]] : <i4> -> <i2>
  // CHECK-NEXT: [[V3:%.+]] = llhd.sig.extract [[V1]] from %idx0 : <i2> -> <i1>
  // CHECK-NEXT: [[V4:%.+]] = llhd.sig.extract [[V2]] from %idx0 : <i2> -> <i1>
  // CHECK-NEXT: [[V5:%.+]] = llhd.sig.extract [[V2]] from %idx1 : <i2> -> <i1>
  // CHECK-NEXT: llhd.drv [[V3]], %in0 after [[TIME]] : i1
  // CHECK-NEXT: llhd.drv [[V4]], %in1 after [[TIME]] : i1
  // CHECK-NEXT: llhd.drv [[V5]], %in0 after [[TIME]] : i1
  // CHECK-NEXT: [[V6:%.+]] = llhd.prb [[OUT]] : i4
  // CHECK-NEXT: hw.output [[V6]] : i4

  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c2_c2 = hw.constant 2 : i2
  %c0_c2 = hw.constant 0 : i2
  %out = llhd.sig %init : i4
  %3 = llhd.sig.extract %out from %c0_c2 : <i4> -> <i2>
  %4 = llhd.sig.extract %out from %c2_c2 : <i4> -> <i2>
  %5 = llhd.sig.extract %3 from %idx0 : <i2> -> <i1>
  %6 = llhd.sig.extract %4 from %idx0 : <i2> -> <i1>
  %7 = llhd.sig.extract %4 from %idx1 : <i2> -> <i1>
  llhd.drv %5, %in0 after %0 : i1
  llhd.drv %6, %in1 after %0 : i1
  llhd.drv %7, %in0 after %0 : i1
  %8 = llhd.prb %out : i4
  hw.output %8 : i4
}

// CHECK-LABEL: @RemoveDriveOnlySignals
hw.module @RemoveDriveOnlySignals(in %d: i42, in %e: i1) {
  %0 = hw.constant 0 : i42
  %1 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NOT: llhd.sig
  %a = llhd.sig %0 : i42
  %b = llhd.sig %0 : i42
  // CHECK-NOT: llhd.drv
  llhd.drv %a, %d after %1 : i42
  llhd.drv %b, %d after %1 if %e : i42
  // CHECK: hw.output
}

// Elements of an array signal are addressed at `index * elementBitWidth`.
// CHECK-LABEL: @ArrayGetProjection
hw.module @ArrayGetProjection(in %in0: i8, in %in1: i8, out o: !hw.array<4xi8>) {
  // CHECK-NOT: llhd.sig
  // CHECK-NOT: llhd.drv
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i2 = hw.constant 0 : i2
  %c2_i2 = hw.constant 2 : i2
  %init = hw.aggregate_constant [0 : i8, 0 : i8, 0 : i8, 0 : i8] : !hw.array<4xi8>
  %sig = llhd.sig %init : !hw.array<4xi8>
  %e0 = llhd.sig.array_get %sig[%c0_i2] : <!hw.array<4xi8>>
  %e2 = llhd.sig.array_get %sig[%c2_i2] : <!hw.array<4xi8>>
  llhd.drv %e0, %in0 after %0 : i8
  llhd.drv %e2, %in1 after %0 : i8
  %prb = llhd.prb %sig : !hw.array<4xi8>
  // Element 0 occupies bits 7:0, so it is injected without a shift. Element 2
  // occupies bits 23:16 and is shifted up by 16.
  // CHECK-DAG: [[C16:%.+]] = hw.constant 16 : i32
  // CHECK-DAG: [[E0:%.+]] = comb.concat {{%.+}}, %in0 : i24, i8
  // CHECK-DAG: [[E2:%.+]] = comb.concat {{%.+}}, %in1 : i24, i8
  // CHECK: [[SHL:%.+]] = comb.shl [[E2]], [[C16]]
  // CHECK: [[OR:%.+]] = comb.or [[SHL]], {{%.+}}
  // CHECK: [[A:%.+]] = hw.bitcast [[OR]] : (i32) -> !hw.array<4xi8>
  // CHECK: hw.output [[A]]
  hw.output %prb : !hw.array<4xi8>
}

// A dynamic array index cannot be scaled by the element width, so the signal
// stays put.
// CHECK-LABEL: @ArrayGetDynamicIndex
hw.module @ArrayGetDynamicIndex(in %idx: i2, in %in0: i8, out o: !hw.array<4xi8>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.aggregate_constant [0 : i8, 0 : i8, 0 : i8, 0 : i8] : !hw.array<4xi8>
  // CHECK: llhd.sig
  %sig = llhd.sig %init : !hw.array<4xi8>
  %e = llhd.sig.array_get %sig[%idx] : <!hw.array<4xi8>>
  // CHECK: llhd.drv
  llhd.drv %e, %in0 after %0 : i8
  %prb = llhd.prb %sig : !hw.array<4xi8>
  hw.output %prb : !hw.array<4xi8>
}

// The first field of a struct occupies the most significant bits, so "a" sits
// at offset 5 and "b" at offset 0.
// CHECK-LABEL: @StructExtractProjection
hw.module @StructExtractProjection(in %av: i3, in %bv: i5, out o: !hw.struct<a: i3, b: i5>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.aggregate_constant [0 : i3, 0 : i5] : !hw.struct<a: i3, b: i5>
  // CHECK-NOT: llhd.sig
  %sig = llhd.sig %init : !hw.struct<a: i3, b: i5>
  %fa = llhd.sig.struct_extract %sig["a"] : <!hw.struct<a: i3, b: i5>>
  %fb = llhd.sig.struct_extract %sig["b"] : <!hw.struct<a: i3, b: i5>>
  // CHECK-NOT: llhd.drv
  llhd.drv %fa, %av after %0 : i3
  llhd.drv %fb, %bv after %0 : i5
  %prb = llhd.prb %sig : !hw.struct<a: i3, b: i5>
  // CHECK-DAG: [[C5:%.+]] = hw.constant 5 : i8
  // CHECK-DAG: [[BPAD:%.+]] = comb.concat {{%.+}}, %bv : i3, i5
  // CHECK: [[APAD:%.+]] = comb.concat {{%.+}}, %av : i5, i3
  // CHECK: [[SHL:%.+]] = comb.shl [[APAD]], [[C5]]
  // CHECK: [[OR:%.+]] = comb.or [[SHL]], {{%.+}}
  // CHECK: [[S:%.+]] = hw.bitcast [[OR]] : (i8) -> !hw.struct<a: i3, b: i5>
  // CHECK: hw.output [[S]]
  hw.output %prb : !hw.struct<a: i3, b: i5>
}

// A struct nested in an array element combines both offsets: element 1 starts
// at bit 6, and "lo" is the last field, so it sits at offset 6.
// CHECK-LABEL: @StructExtractInArray
hw.module @StructExtractInArray(in %v: i2, out o: !hw.array<2xstruct<hi: i4, lo: i2>>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %e = hw.aggregate_constant [0 : i4, 0 : i2] : !hw.struct<hi: i4, lo: i2>
  %init = hw.aggregate_constant [[0 : i4, 0 : i2], [0 : i4, 0 : i2]] : !hw.array<2xstruct<hi: i4, lo: i2>>
  %c1 = hw.constant 1 : i1
  // CHECK-NOT: llhd.sig
  %sig = llhd.sig %init : !hw.array<2xstruct<hi: i4, lo: i2>>
  %elt = llhd.sig.array_get %sig[%c1] : <!hw.array<2xstruct<hi: i4, lo: i2>>>
  %fld = llhd.sig.struct_extract %elt["lo"] : <!hw.struct<hi: i4, lo: i2>>
  // CHECK-NOT: llhd.drv
  llhd.drv %fld, %v after %0 : i2
  %prb = llhd.prb %sig : !hw.array<2xstruct<hi: i4, lo: i2>>
  // CHECK: [[C6:%.+]] = hw.constant 6 : i12
  // CHECK: [[PAD:%.+]] = comb.concat {{%.+}}, %v : i10, i2
  // CHECK: [[SHL:%.+]] = comb.shl [[PAD]], [[C6]]
  // CHECK: hw.output
  hw.output %prb : !hw.array<2xstruct<hi: i4, lo: i2>>
}

// A union's members all share one storage location, which the offset model
// cannot express, so the signal stays put.
// CHECK-LABEL: @UnionExtractUnsupported
hw.module @UnionExtractUnsupported(in %v: i3, out o: i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.constant 0 : i8
  %bc = hw.bitcast %init : (i8) -> !hw.union<a: i3, b: i8>
  // CHECK: llhd.sig
  %sig = llhd.sig %bc : !hw.union<a: i3, b: i8>
  %fa = llhd.sig.struct_extract %sig["a"] : <!hw.union<a: i3, b: i8>>
  // CHECK: llhd.drv
  llhd.drv %fa, %v after %0 : i3
  %prb = llhd.prb %sig : !hw.union<a: i3, b: i8>
  %out = hw.bitcast %prb : (!hw.union<a: i3, b: i8>) -> i8
  hw.output %out : i8
}

// A drive that read-modify-writes the whole struct to update one field is
// narrowed to that field, so it no longer overlaps the sibling drive on "b".
// CHECK-LABEL: @NarrowRmwDrive
hw.module @NarrowRmwDrive(in %av: i2, in %bv: i2, out o: !hw.struct<a: i2, b: i2>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.aggregate_constant [0 : i2, 0 : i2] : !hw.struct<a: i2, b: i2>
  // CHECK-NOT: llhd.sig
  // CHECK-NOT: llhd.drv
  %sig = llhd.sig %init : !hw.struct<a: i2, b: i2>
  %fb = llhd.sig.struct_extract %sig["b"] : <!hw.struct<a: i2, b: i2>>
  llhd.drv %fb, %bv after %0 : i2
  %p = llhd.prb %sig : !hw.struct<a: i2, b: i2>
  %inj = hw.struct_inject %p["a"], %av : !hw.struct<a: i2, b: i2>
  llhd.drv %sig, %inj after %0 : !hw.struct<a: i2, b: i2>
  %prb = llhd.prb %sig : !hw.struct<a: i2, b: i2>
  // CHECK: hw.output
  hw.output %prb : !hw.struct<a: i2, b: i2>
}

// The probe and the drive target need not be spelled the same way: here the
// probe is taken of the whole array and narrowed with a value-level array_get,
// while the drive targets a signal-level projection of the same element.
// CHECK-LABEL: @NarrowRmwDriveThroughArray
hw.module @NarrowRmwDriveThroughArray(in %av: i2, in %bv: i2, out o: !hw.array<2xstruct<a: i2, b: i2>>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.aggregate_constant [[0 : i2, 0 : i2], [0 : i2, 0 : i2]] : !hw.array<2xstruct<a: i2, b: i2>>
  %c1 = hw.constant 1 : i1
  // CHECK-NOT: llhd.sig
  // CHECK-NOT: llhd.drv
  %sig = llhd.sig %init : !hw.array<2xstruct<a: i2, b: i2>>
  %elt = llhd.sig.array_get %sig[%c1] : <!hw.array<2xstruct<a: i2, b: i2>>>
  %fb = llhd.sig.struct_extract %elt["b"] : <!hw.struct<a: i2, b: i2>>
  llhd.drv %fb, %bv after %0 : i2
  %wide = llhd.prb %sig : !hw.array<2xstruct<a: i2, b: i2>>
  %eltval = hw.array_get %wide[%c1] : !hw.array<2xstruct<a: i2, b: i2>>, i1
  %inj = hw.struct_inject %eltval["a"], %av : !hw.struct<a: i2, b: i2>
  llhd.drv %elt, %inj after %0 : !hw.struct<a: i2, b: i2>
  %prb = llhd.prb %sig : !hw.array<2xstruct<a: i2, b: i2>>
  // CHECK: hw.output
  hw.output %prb : !hw.array<2xstruct<a: i2, b: i2>>
}

// A conditional drive really does hold the untouched fields at their old value
// on the cycles it is disabled, so it must not be narrowed. The overlap with
// the drive on "b" then keeps the signal from being promoted.
// CHECK-LABEL: @NoNarrowConditionalRmwDrive
hw.module @NoNarrowConditionalRmwDrive(in %av: i2, in %bv: i2, in %en: i1, out o: !hw.struct<a: i2, b: i2>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.aggregate_constant [0 : i2, 0 : i2] : !hw.struct<a: i2, b: i2>
  // CHECK: llhd.sig
  %sig = llhd.sig %init : !hw.struct<a: i2, b: i2>
  %fb = llhd.sig.struct_extract %sig["b"] : <!hw.struct<a: i2, b: i2>>
  llhd.drv %fb, %bv after %0 : i2
  %p = llhd.prb %sig : !hw.struct<a: i2, b: i2>
  %inj = hw.struct_inject %p["a"], %av : !hw.struct<a: i2, b: i2>
  // CHECK: llhd.drv {{.*}} if
  llhd.drv %sig, %inj after %0 if %en : !hw.struct<a: i2, b: i2>
  %prb = llhd.prb %sig : !hw.struct<a: i2, b: i2>
  hw.output %prb : !hw.struct<a: i2, b: i2>
}

// The injection is rooted at a probe of a different signal, so the drive is a
// genuine whole-struct write rather than a read-modify-write.
// CHECK-LABEL: @NoNarrowForeignProbe
hw.module @NoNarrowForeignProbe(in %av: i2, in %bv: i2, out o: !hw.struct<a: i2, b: i2>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.aggregate_constant [0 : i2, 0 : i2] : !hw.struct<a: i2, b: i2>
  // CHECK: llhd.sig
  %sig = llhd.sig %init : !hw.struct<a: i2, b: i2>
  %other = llhd.sig %init : !hw.struct<a: i2, b: i2>
  %fb = llhd.sig.struct_extract %sig["b"] : <!hw.struct<a: i2, b: i2>>
  llhd.drv %fb, %bv after %0 : i2
  %p = llhd.prb %other : !hw.struct<a: i2, b: i2>
  %inj = hw.struct_inject %p["a"], %av : !hw.struct<a: i2, b: i2>
  // CHECK: llhd.drv
  llhd.drv %sig, %inj after %0 : !hw.struct<a: i2, b: i2>
  %prb = llhd.prb %sig : !hw.struct<a: i2, b: i2>
  hw.output %prb : !hw.struct<a: i2, b: i2>
}

// The outer injection shadows the inner one on the same field. Splitting the
// chain would turn that into two drives fighting over "a", so it is left alone.
// CHECK-LABEL: @NoNarrowShadowedField
hw.module @NoNarrowShadowedField(in %av: i2, in %av2: i2, in %bv: i2, out o: !hw.struct<a: i2, b: i2>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.aggregate_constant [0 : i2, 0 : i2] : !hw.struct<a: i2, b: i2>
  // CHECK: llhd.sig
  %sig = llhd.sig %init : !hw.struct<a: i2, b: i2>
  %fb = llhd.sig.struct_extract %sig["b"] : <!hw.struct<a: i2, b: i2>>
  llhd.drv %fb, %bv after %0 : i2
  %p = llhd.prb %sig : !hw.struct<a: i2, b: i2>
  %inj1 = hw.struct_inject %p["a"], %av : !hw.struct<a: i2, b: i2>
  %inj2 = hw.struct_inject %inj1["a"], %av2 : !hw.struct<a: i2, b: i2>
  // CHECK: llhd.drv
  llhd.drv %sig, %inj2 after %0 : !hw.struct<a: i2, b: i2>
  %prb = llhd.prb %sig : !hw.struct<a: i2, b: i2>
  hw.output %prb : !hw.struct<a: i2, b: i2>
}
