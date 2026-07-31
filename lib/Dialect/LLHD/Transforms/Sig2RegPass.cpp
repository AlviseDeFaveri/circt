//===- Sig2RegPass.cpp - Implement the Sig2Reg Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to promote LLHD signals to SSA values.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-sig2reg"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_SIG2REG
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Represents an offset of an interval relative to a root interval. All values
/// describe number of bits, not elements.
struct Offset {
  Offset(uint64_t min, uint64_t max, ArrayRef<Value> dynamic)
      : min(min), max(max), dynamic(dynamic) {}

  Offset(uint64_t idx) : min(idx), max(idx) {}

  // The lower bound of the offset known statically.
  uint64_t min = 0;
  // The upper bound of the offset known statically.
  uint64_t max = -1;
  // A list of SSA values used to compute the final offset.
  SmallVector<Value> dynamic;

  /// Returns if we know the exact offset statically.
  bool isStatic() const { return min == max; }
};

/// Represents an alias interval within a root interval that is written to or
/// read from. All values refer to number of bits, not elements.
struct Interval {
  Interval(const Offset &low, uint64_t bitwidth, Value value,
           llhd::TimeAttr delay = llhd::TimeAttr())
      : low(low), bitwidth(bitwidth), value(value), delay(delay) {}

  // The offset of the interval relative to the root interval (i.e. all the bits
  // of the original signal).
  Offset low;
  // The width of the interval.
  uint64_t bitwidth;
  // The value written to this interval or the OpResult of a read.
  Value value;
  // The delay with which the value is written.
  llhd::TimeAttr delay;
};

class SigPromoter {
public:
  SigPromoter(llhd::SignalOp sigOp) : sigOp(sigOp) {}

  // Start at the signal operation and traverse all alias operations to compute
  // all the intervals and sort them by ascending offset.
  LogicalResult computeIntervals() {
    SmallVector<std::pair<Operation *, Offset>> stack;

    for (auto *user : sigOp->getUsers())
      stack.emplace_back(user, Offset(0));

    while (!stack.empty()) {
      auto currAndOffset = stack.pop_back_val();
      auto *curr = currAndOffset.first;
      auto offset = currAndOffset.second;

      if (curr->getBlock() != sigOp->getBlock()) {
        LLVM_DEBUG(llvm::dbgs() << "  - User in other block, skipping...\n\n");
        return failure();
      }

      auto result =
          TypeSwitch<Operation *, LogicalResult>(curr)
              .Case<llhd::ProbeOp>([&](llhd::ProbeOp probeOp) {
                auto bw = hw::getBitWidth(probeOp.getResult().getType());
                if (bw <= 0)
                  return failure();

                readIntervals.emplace_back(offset, bw, probeOp.getResult());
                return success();
              })
              .Case<llhd::DriveOp>([&](llhd::DriveOp driveOp) {
                if (driveOp.getEnable()) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  - Conditional driver, skipping...\n\n");
                  return failure();
                }

                auto timeOp =
                    driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
                if (!timeOp) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  - Unknown drive delay, skipping...\n\n");
                  return failure();
                }

                auto bw = hw::getBitWidth(driveOp.getValue().getType());
                if (bw <= 0)
                  return failure();

                intervals.emplace_back(offset, bw, driveOp.getValue(),
                                       timeOp.getValueAttr());
                return success();
              })
              .Case<llhd::SigExtractOp>([&](llhd::SigExtractOp extractOp) {
                if (auto constOp =
                        extractOp.getLowBit().getDefiningOp<hw::ConstantOp>();
                    constOp && offset.isStatic()) {
                  for (auto *user : extractOp->getUsers())
                    stack.emplace_back(
                        user,
                        Offset(constOp.getValue().getZExtValue() + offset.min));

                  return success();
                }

                auto bw = hw::getBitWidth(
                    cast<llhd::RefType>(extractOp.getInput().getType())
                        .getNestedType());
                if (bw <= 0)
                  return failure();

                SmallVector<Value> indices(offset.dynamic);
                indices.push_back(extractOp.getLowBit());

                for (auto *user : extractOp->getUsers())
                  stack.emplace_back(
                      user, Offset(offset.min, offset.max + bw - 1, indices));

                return success();
              })
              .Case<llhd::SigArrayGetOp>([&](llhd::SigArrayGetOp getOp) {
                // Element `i` of an array occupies the bits at offset
                // `i * elementBitWidth`. Only constant indices are supported;
                // a dynamic one would have to be scaled by the element width,
                // which `Offset` cannot express.
                auto constOp =
                    getOp.getIndex().getDefiningOp<hw::ConstantOp>();
                if (!constOp || !offset.isStatic()) {
                  LLVM_DEBUG(llvm::dbgs() << "  - Dynamic array index, "
                                             "skipping...\n\n");
                  return failure();
                }

                auto arrayType = cast<hw::ArrayType>(
                    cast<llhd::RefType>(getOp.getInput().getType())
                        .getNestedType());
                auto elementBw = hw::getBitWidth(arrayType.getElementType());
                if (elementBw <= 0)
                  return failure();

                auto index = constOp.getValue().getZExtValue();
                for (auto *user : getOp->getUsers())
                  stack.emplace_back(user,
                                     Offset(index * elementBw + offset.min));

                return success();
              })
              .Case<llhd::SigStructExtractOp>(
                  [&](llhd::SigStructExtractOp extractOp) {
                    // The first field of a struct occupies the most significant
                    // bits, so a field sits at an offset given by the combined
                    // width of all the fields that follow it.
                    if (!offset.isStatic()) {
                      LLVM_DEBUG(llvm::dbgs() << "  - Dynamic struct offset, "
                                                 "skipping...\n\n");
                      return failure();
                    }

                    // Unions share a single storage location for all their
                    // members, which the offset model cannot express.
                    auto structType = dyn_cast<hw::StructType>(
                        cast<llhd::RefType>(extractOp.getInput().getType())
                            .getNestedType());
                    if (!structType) {
                      LLVM_DEBUG(llvm::dbgs() << "  - Union field access, "
                                                 "skipping...\n\n");
                      return failure();
                    }

                    auto elements = structType.getElements();
                    auto index =
                        structType.getFieldIndex(extractOp.getFieldAttr());
                    if (!index)
                      return failure();

                    uint64_t fieldOffset = 0;
                    for (auto field : elements.drop_front(*index + 1)) {
                      auto bw = hw::getBitWidth(field.type);
                      if (bw < 0)
                        return failure();
                      fieldOffset += bw;
                    }

                    for (auto *user : extractOp->getUsers())
                      stack.emplace_back(user,
                                         Offset(fieldOffset + offset.min));

                    return success();
                  })
              .Default([](auto *op) {
                LLVM_DEBUG(llvm::dbgs() << "  - User that is not a probe or "
                                           "drive, skipping...\n    "
                                        << *op << "\n\n");
                return failure();
              });

      if (failed(result))
        return failure();

      toDelete.push_back(curr);
    }

    llvm::sort(intervals, [](const Interval &a, const Interval &b) {
      return a.low.min < b.low.min;
    });

    LLVM_DEBUG({
      llvm::dbgs() << "  - Detected intervals:\n";
      dumpIntervals(llvm::dbgs(), 4);
    });

    return success();
  }

#ifndef NDEBUG

  /// Print the list of intervals in a readable format for debugging.
  void dumpIntervals(llvm::raw_ostream &os, unsigned indent = 0) {
    os << llvm::indent(indent) << "[\n";
    for (const auto &interval : intervals) {
      os << llvm::indent(indent + 2) << "<from [" << interval.low.min << ", "
         << interval.low.max << "]\n";
      os << llvm::indent(indent + 3) << "width " << interval.bitwidth << "\n";

      for (auto idx : interval.low.dynamic)
        os << llvm::indent(indent + 3) << idx << "\n";

      os << llvm::indent(indent + 3) << "value: " << interval.value << "\n";
      os << llvm::indent(indent + 3) << "delay: " << interval.delay << "\n";
      os << llvm::indent(indent + 2) << ">,\n";
    }
    os << llvm::indent(indent) << "]\n";
  }

#endif

  /// Check if we can promote the entire signal according to the current
  /// limitations of the pass.
  bool isPromotable() {
    for (unsigned i = 0; i < intervals.size(); ++i) {
      if (i >= intervals.size() - 1)
        break;

      if (intervals[i].low.max + intervals[i].bitwidth - 1 >
          intervals[i + 1].low.min) {
        LLVM_DEBUG({
          llvm::dbgs() << "  - Potentially overlapping drives, skipping...\n\n";
        });
        return false;
      }
    }

    return true;
  }

  /// Promote the signal. This builds the necessary operations, replaces the
  /// values, and removes the signal and signal value handling operations.
  void promote() {
    auto bw = hw::getBitWidth(sigOp.getInit().getType());
    assert(bw > 0 && "bw must be known and non-zero");

    OpBuilder builder(sigOp);
    Value val = sigOp.getInit();
    Location loc = sigOp->getLoc();
    auto type = builder.getIntegerType(bw);
    val = builder.createOrFold<hw::BitcastOp>(loc, type, val);

    // Handle the writes by starting with the signal init value and injecting
    // the written values at the right offsets.
    for (auto interval : intervals) {
      Value invMask = hw::ConstantOp::create(
          builder, loc, APInt::getAllOnes(interval.bitwidth));

      if (uint64_t(bw) > interval.bitwidth) {
        Value pad = hw::ConstantOp::create(
            builder, loc, APInt::getZero(bw - interval.bitwidth));
        invMask = builder.createOrFold<comb::ConcatOp>(loc, pad, invMask);
      }

      Value amt = buildDynamicIndex(builder, loc, interval.low.min,
                                    interval.low.dynamic, bw);
      invMask = builder.createOrFold<comb::ShlOp>(loc, invMask, amt);
      Value allOnes =
          hw::ConstantOp::create(builder, loc, APInt::getAllOnes(bw));
      Value mask = builder.createOrFold<comb::XorOp>(loc, invMask, allOnes);
      val = builder.createOrFold<comb::AndOp>(loc, val, mask);

      Value assignVal = builder.createOrFold<hw::BitcastOp>(
          loc, builder.getIntegerType(interval.bitwidth), interval.value);

      if (uint64_t(bw) > interval.bitwidth) {
        Value pad = hw::ConstantOp::create(
            builder, loc, APInt::getZero(bw - interval.bitwidth));
        assignVal = builder.createOrFold<comb::ConcatOp>(loc, pad, assignVal);
      }

      assignVal = builder.createOrFold<comb::ShlOp>(loc, assignVal, amt);
      if (!isImmediate(interval.delay))
        assignVal =
            builder.createOrFold<llhd::DelayOp>(loc, assignVal, interval.delay);
      val = builder.createOrFold<comb::OrOp>(loc, assignVal, val);
    }

    // Handle the reads by extracting right number of bits at the right offset.
    for (auto interval : readIntervals) {
      if (interval.low.isStatic()) {
        Value read = builder.createOrFold<comb::ExtractOp>(
            loc, builder.getIntegerType(interval.bitwidth), val,
            interval.low.min);
        read = builder.createOrFold<hw::BitcastOp>(
            loc, interval.value.getType(), read);
        if (read != interval.value) {
          interval.value.replaceAllUsesWith(read);
        }
        continue;
      }

      Value read = buildDynamicIndex(builder, loc, interval.low.min,
                                     interval.low.dynamic, bw);
      read = builder.createOrFold<comb::ShrUOp>(loc, val, read);
      read = builder.createOrFold<comb::ExtractOp>(
          loc, builder.getIntegerType(interval.bitwidth), read, 0);
      read = builder.createOrFold<hw::BitcastOp>(loc, interval.value.getType(),
                                                 read);
      if (read != interval.value) {
        interval.value.replaceAllUsesWith(read);
      }
    }

    // Delete all operations operating on signal values.
    for (auto *op : llvm::reverse(toDelete))
      op->erase();

    sigOp->erase();
  }

private:
  /// Given a static offset and a list of dynamic offset values, materialize an
  /// SSA value that adds all these offsets together and is an integer with the
  /// given 'width'.
  Value buildDynamicIndex(OpBuilder &builder, Location loc,
                          uint64_t constOffset, ArrayRef<Value> indices,
                          uint64_t width) {
    Value index = hw::ConstantOp::create(
        builder, loc, builder.getIntegerType(width), constOffset);

    for (auto idx : indices) {
      auto bw = hw::getBitWidth(idx.getType());
      Value pad =
          hw::ConstantOp::create(builder, loc, APInt::getZero(width - bw));
      idx = builder.createOrFold<comb::ConcatOp>(loc, pad, idx);
      index = builder.createOrFold<comb::AddOp>(loc, index, idx);
    }

    return index;
  }

  bool isImmediate(llhd::TimeAttr attr) const {
    return attr.getTime() == 0 && attr.getDelta() == 0 &&
           attr.getEpsilon() == 1;
  }

  // The signal to be promoted.
  llhd::SignalOp sigOp;
  // Intervals written to.
  SmallVector<Interval> intervals;
  // Intervals read from.
  SmallVector<Interval> readIntervals;
  // Operations to delete after promotion is done.
  SmallVector<Operation *> toDelete;
};

/// One step of a projection into an aggregate: either a named struct field or
/// an array index. Indices compare by SSA value, which is conservative but
/// exact enough once CSE has run.
struct AccessStep {
  StringAttr field;
  Value index;

  bool operator==(const AccessStep &other) const {
    return field == other.field && index == other.index;
  }
};

/// Peel the chain of signal projections feeding `ref` and return the root
/// signal, recording in `path` the steps that lead from the root back to `ref`.
static Value getRefProjectionPath(Value ref, SmallVectorImpl<AccessStep> &path) {
  auto start = path.size();
  while (auto *op = ref.getDefiningOp()) {
    if (auto getOp = dyn_cast<llhd::SigArrayGetOp>(op)) {
      path.push_back({{}, getOp.getIndex()});
      ref = getOp.getInput();
      continue;
    }
    if (auto extractOp = dyn_cast<llhd::SigStructExtractOp>(op)) {
      path.push_back({extractOp.getFieldAttr(), {}});
      ref = extractOp.getInput();
      continue;
    }
    break;
  }
  std::reverse(path.begin() + start, path.end());
  return ref;
}

/// The same for the value-level projections feeding `value`, which is how a
/// probe of a whole aggregate gets narrowed down to one of its elements.
static Value getValueProjectionPath(Value value,
                                    SmallVectorImpl<AccessStep> &path) {
  auto start = path.size();
  while (auto *op = value.getDefiningOp()) {
    if (auto getOp = dyn_cast<hw::ArrayGetOp>(op)) {
      path.push_back({{}, getOp.getIndex()});
      value = getOp.getInput();
      continue;
    }
    if (auto extractOp = dyn_cast<hw::StructExtractOp>(op)) {
      path.push_back({extractOp.getFieldNameAttr(), {}});
      value = extractOp.getInput();
      continue;
    }
    break;
  }
  std::reverse(path.begin() + start, path.end());
  return value;
}

/// Rewrite drives that read-modify-write an aggregate to drive only the fields
/// they actually change:
///
/// ```
/// %p = llhd.prb %ref
/// %v = hw.struct_inject %p["f"], %x
/// llhd.drv %ref, %v after %t
/// ```
///
/// becomes
///
/// ```
/// %f = llhd.sig.struct_extract %ref["f"]
/// llhd.drv %f, %x after %t
/// ```
///
/// Frontends produce the wide form for an `always_comb` that assigns a single
/// field, since the untouched fields have to be carried over from the probe.
/// That makes the drive span the whole aggregate and overlap any sibling drives
/// on the other fields, which stops `SigPromoter` from promoting the signal.
/// The narrow form reaches the same fixpoint -- re-driving a field with the
/// value just probed off it is a no-op -- without the spurious overlap.
static void narrowReadModifyWriteDrives(hw::HWModuleOp moduleOp) {
  SmallVector<llhd::DriveOp> driveOps(moduleOp.getOps<llhd::DriveOp>());

  for (auto driveOp : driveOps) {
    // A conditional drive leaves the aggregate untouched on some cycles, so the
    // carried-over fields are load-bearing rather than redundant.
    if (driveOp.getEnable())
      continue;

    // Peel off the chain of injections and check that it bottoms out in a probe
    // of the very signal being driven.
    SmallVector<hw::StructInjectOp> injectOps;
    Value current = driveOp.getValue();
    while (auto injectOp = current.getDefiningOp<hw::StructInjectOp>()) {
      injectOps.push_back(injectOp);
      current = injectOp.getInput();
    }
    if (injectOps.empty())
      continue;

    // The injected-into value has to be the very storage being driven. Both
    // sides may be projections of a larger aggregate, and they need not be
    // spelled the same way: the drive typically targets a chain of signal
    // projections, while the probe is taken of the whole signal and narrowed
    // down with value projections. Compare the two as paths from a common root.
    SmallVector<AccessStep> refPath, probePath;
    Value root = getRefProjectionPath(driveOp.getSignal(), refPath);

    Value probed = getValueProjectionPath(current, probePath);
    auto probeOp = probed.getDefiningOp<llhd::ProbeOp>();
    if (!probeOp)
      continue;

    SmallVector<AccessStep> probeRefPath;
    Value probeRoot = getRefProjectionPath(probeOp.getSignal(), probeRefPath);
    probeRefPath.append(probePath);

    if (probeRoot != root || probeRefPath != refPath)
      continue;

    // Everything has to sit in the drive's block so the projections we create
    // stay in scope and keep their position relative to the other drives.
    if (probeOp->getBlock() != driveOp->getBlock() ||
        llvm::any_of(injectOps, [&](auto injectOp) {
          return injectOp->getBlock() != driveOp->getBlock();
        }))
      continue;

    // An outer injection shadows an inner one on the same field. Splitting the
    // chain would turn that into two drives fighting over one field, so leave
    // the drive alone.
    SmallPtrSet<Attribute, 4> seenFields;
    if (llvm::any_of(injectOps, [&](auto injectOp) {
          return !seenFields.insert(injectOp.getFieldNameAttr()).second;
        }))
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "  - Narrowing read-modify-write drive " << driveOp << "\n");

    OpBuilder builder(driveOp);
    for (auto injectOp : injectOps) {
      Value fieldRef = llhd::SigStructExtractOp::create(
          builder, injectOp.getLoc(), driveOp.getSignal(),
          injectOp.getFieldNameAttr());
      llhd::DriveOp::create(builder, driveOp.getLoc(), fieldRef,
                            injectOp.getNewValue(), driveOp.getTime(), Value());
    }
    driveOp.erase();

    // The injections and the probe are dead unless something else reads them.
    for (auto injectOp : llvm::reverse(injectOps))
      if (injectOp->use_empty())
        injectOp->erase();
    if (probeOp->use_empty())
      probeOp->erase();
  }
}

struct Sig2RegPass : public circt::llhd::impl::Sig2RegBase<Sig2RegPass> {
  void runOnOperation() override;
};
} // namespace

void Sig2RegPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "=== Sig2Reg in module " << moduleOp.getSymName()
                          << "\n\n");

  narrowReadModifyWriteDrives(moduleOp);

  for (auto sigOp :
       llvm::make_early_inc_range(moduleOp.getOps<llhd::SignalOp>())) {
    // If the signal is only driven, but never read or used otherwise, remove
    // it.
    if (llvm::all_of(sigOp->getUses(), [](auto &use) {
          return isa<llhd::DriveOp>(use.getOwner()) &&
                 use.getOperandNumber() == 0;
        })) {
      LLVM_DEBUG(llvm::dbgs() << "  - Removing drive-only signal "
                              << sigOp.getName() << "\n");
      for (auto *user : llvm::make_early_inc_range(sigOp->getUsers()))
        user->erase();
      sigOp.erase();
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "  - Attempting to promote signal "
                            << sigOp.getName() << "\n");
    SigPromoter promoter(sigOp);
    if (failed(promoter.computeIntervals()) || !promoter.isPromotable())
      continue;

    promoter.promote();
    LLVM_DEBUG(llvm::dbgs() << "  - Successfully promoted!\n\n");
  }

  LLVM_DEBUG({
    if (moduleOp.getOps<llhd::SignalOp>().empty())
      llvm::dbgs() << "  Successfully promoted all signals in module!\n";
  });

  LLVM_DEBUG(llvm::dbgs() << "\n");
}
