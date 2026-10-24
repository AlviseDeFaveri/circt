//===- CompRegToFirReg.cpp - seq.compreg to seq.firreg --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_COMPREGTOFIRREG
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace seq;

namespace {
struct CompRegToFirRegPass
    : public circt::seq::impl::CompRegToFirRegBase<CompRegToFirRegPass> {
  void runOnOperation() override;
};
} // namespace

void CompRegToFirRegPass::runOnOperation() {
  SmallSetVector<InitialOp, 4> initialOps;
  for (auto regOp :
       llvm::make_early_inc_range(getOperation().getOps<CompRegOp>())) {
    // A `seq.firreg` can only be preset to a constant integer.
    IntegerAttr preset;
    if (auto initialValue = regOp.getInitialValue()) {
      if (!matchPattern(unwrapImmutableValue(initialValue),
                        m_Constant(&preset)))
        continue;
      initialOps.insert(initialValue.getDefiningOp<InitialOp>());
    }

    OpBuilder builder(regOp);
    auto name = regOp.getNameAttr();
    if (!name)
      name = builder.getStringAttr("");
    FirRegOp firRegOp;
    if (regOp.getReset())
      firRegOp = FirRegOp::create(
          builder, regOp.getLoc(), regOp.getInput(), regOp.getClk(), name,
          regOp.getReset(), regOp.getResetValue(), regOp.getInnerSymAttr(),
          /*isAsync=*/false, preset);
    else
      firRegOp = FirRegOp::create(builder, regOp.getLoc(), regOp.getInput(),
                                  regOp.getClk(), name, regOp.getInnerSymAttr(),
                                  preset);
    regOp.replaceAllUsesWith(firRegOp.getResult());
    regOp.erase();
  }

  // Remove initial ops that are no longer used by any register.
  for (auto initialOp : initialOps)
    if (initialOp->use_empty())
      initialOp.erase();
}
