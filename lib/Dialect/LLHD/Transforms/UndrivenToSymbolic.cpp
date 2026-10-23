//===- UndrivenToSymbolic.cpp - Replace undriven signals with symbols -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-undriven-to-symbolic"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_UNDRIVENTOSYMBOLICPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

/// Collect the probes and projections of an undriven signal. Returns false if
/// the signal has any other kind of user, such as a drive or an op passing the
/// signal on to a place where it may be driven.
static bool collectUsers(SignalOp sigOp, SmallVectorImpl<ProbeOp> &probes,
                         SmallVectorImpl<Operation *> &projections) {
  SmallVector<Value> worklist = {sigOp.getResult()};
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (auto probeOp = dyn_cast<ProbeOp>(user)) {
        probes.push_back(probeOp);
        continue;
      }
      if (isa<SigExtractOp, SigArrayGetOp, SigArraySliceOp, SigStructExtractOp>(
              user) &&
          use.getOperandNumber() == 0) {
        projections.push_back(user);
        worklist.push_back(user->getResult(0));
        continue;
      }
      return false;
    }
  }
  return true;
}

/// Compute the value probed from `signal`, a projection chain rooted in the
/// signal whose value is `rootValue`.
static Value getProjectedValue(OpBuilder &builder, Value signal,
                               Value rootValue) {
  auto *op = signal.getDefiningOp();
  if (isa<SignalOp>(op))
    return rootValue;
  auto value = getProjectedValue(builder, op->getOperand(0), rootValue);
  return TypeSwitch<Operation *, Value>(op)
      .Case<SigArrayGetOp>([&](auto op) {
        return builder.createOrFold<hw::ArrayGetOp>(op.getLoc(), value,
                                                    op.getIndex());
      })
      .Case<SigArraySliceOp>([&](auto op) {
        return builder.createOrFold<hw::ArraySliceOp>(
            op.getLoc(), cast<RefType>(op.getType()).getNestedType(), value,
            op.getLowIndex());
      })
      .Case<SigStructExtractOp>([&](auto op) {
        // SigStructExtractOp is used for both struct and union member access.
        if (isa<hw::UnionType>(value.getType()))
          return builder.createOrFold<hw::UnionExtractOp>(op.getLoc(), value,
                                                          op.getFieldAttr());
        return builder.createOrFold<hw::StructExtractOp>(op.getLoc(), value,
                                                         op.getFieldAttr());
      })
      .Case<SigExtractOp>([&](auto op) {
        auto type = cast<RefType>(op.getType()).getNestedType();
        return comb::createDynamicExtract(builder, op.getLoc(), value,
                                          op.getLowBit(),
                                          type.getIntOrFloatBitWidth());
      });
}

namespace {
struct UndrivenToSymbolicPass
    : public llhd::impl::UndrivenToSymbolicPassBase<UndrivenToSymbolicPass> {
  void runOnOperation() override;
};
} // namespace

void UndrivenToSymbolicPass::runOnOperation() {
  for (auto sigOp :
       llvm::make_early_inc_range(getOperation().getOps<SignalOp>())) {
    if (sigOp.getInit())
      continue;
    SmallVector<ProbeOp> probes;
    SmallVector<Operation *> projections;
    if (!collectUsers(sigOp, probes, projections))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Replacing undriven " << sigOp << "\n");

    OpBuilder builder(sigOp);
    Value symbol = verif::SymbolicValueOp::create(
        builder, sigOp.getLoc(), sigOp.getType().getNestedType(),
        sigOp.getNameAttr());
    for (auto probeOp : probes) {
      builder.setInsertionPoint(probeOp);
      probeOp.replaceAllUsesWith(
          getProjectedValue(builder, probeOp.getSignal(), symbol));
      probeOp.erase();
    }
    for (auto *op : llvm::reverse(projections))
      op->erase();
    sigOp.erase();
  }
}
