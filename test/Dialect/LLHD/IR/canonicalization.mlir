// RUN: circt-opt --canonicalize %s | FileCheck %s

// A drive whose value writes most of the signal straight back from a probe of
// that same signal is narrowed to the part that actually changes.

// CHECK-LABEL: @NarrowConcatDrive
hw.module @NarrowConcatDrive(in %x: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  %prb = llhd.prb %sig : i8
  %hi = comb.extract %prb from 4 : (i8) -> i4
  %lo = comb.extract %prb from 0 : (i8) -> i3
  %val = comb.concat %hi, %x, %lo : i4, i1, i3
  // CHECK: [[C3:%.+]] = hw.constant 3 : i3
  // CHECK: [[BIT:%.+]] = llhd.sig.extract %sig from [[C3]] : <i8> -> <i1>
  // CHECK: llhd.drv [[BIT]], %x
  llhd.drv %sig, %val after %0 : i8
}

// The enable is carried over to the narrowed drive.
// CHECK-LABEL: @NarrowConcatDriveWithEnable
hw.module @NarrowConcatDriveWithEnable(in %x: i1, in %en: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  %prb = llhd.prb %sig : i8
  %hi = comb.extract %prb from 4 : (i8) -> i4
  %lo = comb.extract %prb from 0 : (i8) -> i3
  %val = comb.concat %hi, %x, %lo : i4, i1, i3
  // CHECK: llhd.drv {{%.+}}, %x after {{%.+}} if %en : i1
  llhd.drv %sig, %val after %0 if %en : i8
}

// Injecting into a probed array narrows to a drive of that element, and the
// concat inside the element narrows further.
// CHECK-LABEL: @NarrowArrayInjectDrive
hw.module @NarrowArrayInjectDrive(in %x: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c1_i2 = hw.constant 1 : i2
  %init = hw.aggregate_constant [0 : i8, 0 : i8, 0 : i8, 0 : i8] : !hw.array<4xi8>
  %sig = llhd.sig %init : !hw.array<4xi8>
  %prb = llhd.prb %sig : !hw.array<4xi8>
  %elem = hw.array_get %prb[%c1_i2] : !hw.array<4xi8>, i2
  %hi = comb.extract %elem from 4 : (i8) -> i4
  %lo = comb.extract %elem from 0 : (i8) -> i3
  %val = comb.concat %hi, %x, %lo : i4, i1, i3
  %inj = hw.array_inject %prb[%c1_i2], %val : !hw.array<4xi8>, i2
  // CHECK: [[ELEM:%.+]] = llhd.sig.array_get %sig[%c1_i2]
  // CHECK: [[BIT:%.+]] = llhd.sig.extract [[ELEM]] from {{%.+}} : <i8> -> <i1>
  // CHECK: llhd.drv [[BIT]], %x
  llhd.drv %sig, %inj after %0 : !hw.array<4xi8>
}

// A drive that changes nothing is left alone; removing it would drop a driver.
// CHECK-LABEL: @NoOpDriveIsKept
hw.module @NoOpDriveIsKept() {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  %prb = llhd.prb %sig : i8
  %hi = comb.extract %prb from 4 : (i8) -> i4
  %lo = comb.extract %prb from 0 : (i8) -> i4
  %val = comb.concat %hi, %lo : i4, i4
  // CHECK: llhd.drv %sig, {{%.+}} : i8
  llhd.drv %sig, %val after %0 : i8
}

// Slices read from a different signal are not write-backs.
// CHECK-LABEL: @OtherSignalIsNotWriteBack
hw.module @OtherSignalIsNotWriteBack(in %x: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  %other = llhd.sig %c0_i8 : i8
  %prb = llhd.prb %other : i8
  %hi = comb.extract %prb from 4 : (i8) -> i4
  %lo = comb.extract %prb from 0 : (i8) -> i3
  %val = comb.concat %hi, %x, %lo : i4, i1, i3
  // CHECK: llhd.drv %sig, {{%.+}} : i8
  llhd.drv %sig, %val after %0 : i8
}

// Inside a process a probe may predate a wait, so the write-back is meaningful
// and the drive must not be narrowed.
// CHECK-LABEL: @NoNarrowingInsideProcess
hw.module @NoNarrowingInsideProcess(in %x: i1) {
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  llhd.process {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %prb = llhd.prb %sig : i8
    llhd.wait ^bb1
  ^bb1:
    %hi = comb.extract %prb from 4 : (i8) -> i4
    %lo = comb.extract %prb from 0 : (i8) -> i3
    %val = comb.concat %hi, %x, %lo : i4, i1, i3
    // CHECK: llhd.drv %sig, {{%.+}} : i8
    llhd.drv %sig, %val after %0 : i8
    llhd.halt
  }
}
