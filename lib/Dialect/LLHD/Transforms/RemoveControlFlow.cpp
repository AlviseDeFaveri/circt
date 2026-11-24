//===----------------------------------------------------------------------===//
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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-remove-control-flow"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_REMOVECONTROLFLOWPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {
/// A helper struct that tracks a boolean condition as either a constant false,
/// constant true, or an SSA value.
struct Condition {
  Condition() {}
  Condition(Value value) : pair(value, 0) {
    if (value) {
      if (matchPattern(value, m_One()))
        *this = Condition(true);
      if (matchPattern(value, m_Zero()))
        *this = Condition(false);
    }
  }
  Condition(bool konst) : pair(nullptr, konst ? 1 : 2) {}

  explicit operator bool() const {
    return pair.getPointer() != nullptr || pair.getInt() != 0;
  }

  bool isTrue() const { return !pair.getPointer() && pair.getInt() == 1; }
  bool isFalse() const { return !pair.getPointer() && pair.getInt() == 2; }
  Value getValue() const { return pair.getPointer(); }

  /// Turn this condition into an SSA value, creating an `hw.constant` if the
  /// condition is a constant.
  Value materialize(OpBuilder &builder, Location loc) const {
    if (isTrue())
      return hw::ConstantOp::create(builder, loc, APInt(1, 1));
    if (isFalse())
      return hw::ConstantOp::create(builder, loc, APInt(1, 0));
    return pair.getPointer();
  }

  Condition orWith(Condition other, OpBuilder &builder) const {
    if (isTrue() || other.isTrue())
      return true;
    if (isFalse())
      return other;
    if (other.isFalse())
      return *this;
    return builder.createOrFold<comb::OrOp>(getValue().getLoc(), getValue(),
                                            other.getValue());
  }

  Condition andWith(Condition other, OpBuilder &builder) const {
    if (isFalse() || other.isFalse())
      return false;
    if (isTrue())
      return other;
    if (other.isTrue())
      return *this;
    return builder.createOrFold<comb::AndOp>(getValue().getLoc(), getValue(),
                                             other.getValue());
  }

  Condition inverted(OpBuilder &builder) const {
    if (isTrue())
      return false;
    if (isFalse())
      return true;
    return comb::createOrFoldNot(builder, getValue().getLoc(), getValue());
  }

private:
  llvm::PointerIntPair<Value, 2> pair;
};
} // namespace

/// Compute the branch decisions that cause control to flow from the dominator
/// to the target block.
///
/// TODO: This eagerly aggregates all control flow decisions. It may be more
/// efficient to first determine which blocks lie in between dominator and
/// target, and then only check that we are not taking decisions that cause
/// control flow to *leave* that set of blocks.
static Condition getBranchDecisionsFromDominatorToTarget(
    OpBuilder &builder, Block *dominator, Block *target,
    SmallDenseMap<std::pair<Block *, Block *>, Condition> &decisions) {
  if (auto decision = decisions.lookup({dominator, target}))
    return decision;

  SmallPtrSet<Block *, 8> visitedBlocks;
  visitedBlocks.insert(dominator); // stop at the dominator
  if (auto &decision = decisions[{dominator, dominator}]; !decision)
    decision = Condition(true);

  // Traverse the blocks in inverse post order. This ensures that we are
  // visiting all of a block's predecessors before we visit the block itself.
  // This allows us to first compute the decision leading control flow to each
  // of the predecessors, such that the current block can then just combine the
  // predecessor decisions.
  for (auto *block : llvm::inverse_post_order_ext(target, visitedBlocks)) {
    auto merged = Condition(false);
    for (auto *pred : block->getPredecessors()) {
      auto predDecision = decisions.lookup({dominator, pred});
      assert(predDecision);
      if (pred->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = cast<cf::CondBranchOp>(pred->getTerminator());
        if (condBr.getTrueDest() == condBr.getFalseDest()) {
          merged = merged.orWith(predDecision, builder);
        } else {
          auto cond = Condition(condBr.getCondition());
          if (condBr.getFalseDest() == block)
            cond = cond.inverted(builder);
          merged = merged.orWith(cond.andWith(predDecision, builder), builder);
        }
      } else {
        merged = merged.orWith(predDecision, builder);
      }
    }
    assert(merged);
    decisions.insert({{dominator, block}, merged});
  }

  return decisions.lookup({dominator, target});
}

//===----------------------------------------------------------------------===//
// Control Flow Removal
//===----------------------------------------------------------------------===//

namespace {
/// The main helper struct implementing control flow removal for a region.
struct CFRemover {
  CFRemover(Region &region) : region(region) {}
  void run();

  /// The region within which we are removing control flow.
  Region &region;
  /// The blocks in the region, sorted such that a block's predecessors appear
  /// in the list before the block itself.
  SmallVector<Block *> sortedBlocks;
  /// The dominance information for the region.
  DominanceInfo domInfo;
};
} // namespace

void CFRemover::run() {
  LLVM_DEBUG(llvm::dbgs() << "Removing control flow in " << region.getLoc()
                          << "\n");

  // Establish a topological order of the blocks in the region. Give up if we
  // detect a control flow cycle. Also take note of all YieldOps, such that we
  // can combine them into a single yield block later.
  SmallVector<YieldOp, 2> yieldOps;
  SmallPtrSet<Block *, 8> visitedBlocks, ipoSet;
  for (auto &block : region) {
    for (auto *ipoBlock : llvm::inverse_post_order_ext(&block, ipoSet)) {
      if (!llvm::all_of(ipoBlock->getPredecessors(), [&](auto *pred) {
            return visitedBlocks.contains(pred);
          })) {
        LLVM_DEBUG(llvm::dbgs() << "- Loop detected, giving up\n");
        return;
      }
      visitedBlocks.insert(ipoBlock);
      sortedBlocks.push_back(ipoBlock);
    }

    // Give up if there are any side-effecting ops in the region.
    for (auto &op : block) {
      if (!isMemoryEffectFree(&op)) {
        LLVM_DEBUG(llvm::dbgs() << "- Has side effects, giving up\n");
        return;
      }
    }

    // Check that we know what to do with all terminators.
    if (!isa<YieldOp, cf::BranchOp, cf::CondBranchOp>(block.getTerminator())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Has unsupported terminator "
                 << block.getTerminator()->getName() << ", giving up\n");
      return;
    }

    // Keep track of yield ops.
    if (auto yieldOp = dyn_cast<YieldOp>(block.getTerminator()))
      yieldOps.push_back(yieldOp);
  }

  // If there are multiple yield ops, factor them out into a single yield block.
  auto yieldOp = yieldOps[0];
  if (yieldOps.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "- Creating single yield block\n");
    OpBuilder builder(region.getContext());
    SmallVector<Location> locs(yieldOps[0].getNumOperands(), region.getLoc());
    auto *yieldBlock = builder.createBlock(&region, region.end(),
                                           yieldOps[0].getOperandTypes(), locs);
    sortedBlocks.push_back(yieldBlock);
    yieldOp =
        YieldOp::create(builder, region.getLoc(), yieldBlock->getArguments());
    for (auto yieldOp : yieldOps) {
      builder.setInsertionPoint(yieldOp);
      cf::BranchOp::create(builder, yieldOp.getLoc(), yieldBlock,
                           yieldOp.getOperands());
      yieldOp.erase();
    }
  }

  // Compute the dominance info for this region.
  domInfo = DominanceInfo(region.getParentOp());

  // Move operations into the entry block, replacing block arguments with
  // multiplexers as we go. The block order guarantees that we visit a block's
  // predecessors before we visit the block itself.
  SmallDenseMap<std::pair<Block *, Block *>, Condition> decisionCache;
  auto *entryBlock = sortedBlocks.front();
  for (auto *block : sortedBlocks) {
    if (!domInfo.isReachableFromEntry(block))
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "- Merging block ";
      block->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Find the nearest common dominator block of all predecessors. Any block
    // arguments reaching the current block will only depend on control flow
    // decisions between this dominator block and the current block.
    auto *domBlock = block;
    for (auto *pred : block->getPredecessors())
      if (domInfo.isReachableFromEntry(pred))
        domBlock = domInfo.findNearestCommonDominator(domBlock, pred);
    LLVM_DEBUG({
      llvm::dbgs() << "  - Common dominator: ";
      domBlock->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Convert the block arguments into multiplexers.
    SmallVector<Block *, 4> reachablePreds;
    for (auto *pred : block->getPredecessors())
      if (domInfo.isReachableFromEntry(pred))
        reachablePreds.push_back(pred);

    llvm::stable_sort(reachablePreds, [&](Block *a, Block *b) {
      return a == domBlock && b != domBlock;
    });

    OpBuilder builder(entryBlock->getTerminator());

    // Collect the values flowing into the block from each predecessor, along
    // with the branch condition under which control flows along that edge.
    struct Incoming {
      Block *pred;
      SmallVector<Value> args;
      // The branch condition guarding the edge, or null if unconditional.
      Value cond;
      bool invCond = false;
      // The full decision leading from `domBlock` along this edge. Computed
      // lazily, since edges carrying the base value don't need it.
      Condition decision;
    };
    SmallVector<Incoming, 4> incomings;
    SmallPtrSet<Block *, 4> seenPreds;
    for (auto *pred : reachablePreds) {
      // A block may be listed multiple times in the predecessors.
      if (!seenPreds.insert(pred).second)
        continue;

      // Only consider values coming from reachable predecessors.
      if (!domInfo.isReachableFromEntry(pred))
        continue;

      auto &incoming = incomings.emplace_back();
      incoming.pred = pred;

      // Handle the different terminators that we support.
      if (auto condBrOp = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (condBrOp.getTrueDest() == condBrOp.getFalseDest()) {
          // Both destinations lead to the current block. Insert a mux to
          // collapse the destination operands and then treat this as an
          // unconditional branch to the current block.
          LLVM_DEBUG(llvm::dbgs() << "  - Both from " << condBrOp << "\n");
          incoming.args.reserve(block->getNumArguments());
          for (auto [trueArg, falseArg] :
               llvm::zip(condBrOp.getTrueDestOperands(),
                         condBrOp.getFalseDestOperands())) {
            incoming.args.push_back(builder.createOrFold<comb::MuxOp>(
                trueArg.getLoc(), condBrOp.getCondition(), trueArg, falseArg));
          }
        } else if (condBrOp.getTrueDest() == block) {
          // The branch leads to the current block if the condition is true.
          LLVM_DEBUG(llvm::dbgs() << "  - True from " << condBrOp << "\n");
          incoming.args = llvm::to_vector(condBrOp.getTrueDestOperands());
          incoming.cond = condBrOp.getCondition();
        } else {
          // The branch leads to the current block if the condition is false.
          LLVM_DEBUG(llvm::dbgs() << "  - False from " << condBrOp << "\n");
          incoming.args = llvm::to_vector(condBrOp.getFalseDestOperands());
          incoming.cond = condBrOp.getCondition();
          incoming.invCond = true;
        }
      } else {
        auto brOp = cast<cf::BranchOp>(pred->getTerminator());
        LLVM_DEBUG(llvm::dbgs() << "  - From " << brOp << "\n");
        incoming.args = llvm::to_vector(brOp.getDestOperands());
      }
    }

    // Compute the decision that causes control to flow from `domBlock` into
    // the current block along the given incoming edge.
    auto getDecision = [&](Incoming &incoming) {
      if (!incoming.decision) {
        auto decision = getBranchDecisionsFromDominatorToTarget(
            builder, domBlock, incoming.pred, decisionCache);
        if (incoming.cond) {
          auto cond = Condition(incoming.cond);
          if (incoming.invCond)
            cond = cond.inverted(builder);
          decision = decision.andWith(cond, builder);
        }
        incoming.decision = decision;
      }
      return incoming.decision;
    };

    // Determine the blocks through which control can flow from `domBlock` into
    // the current block, in topological order. Since `domBlock` dominates all
    // predecessors, walking backwards from them always ends at `domBlock`.
    SmallDenseMap<Block *, unsigned> incomingIndices;
    for (auto [idx, incoming] : llvm::enumerate(incomings))
      incomingIndices.insert({incoming.pred, idx});
    SmallPtrSet<Block *, 16> betweenSet;
    SmallVector<Block *> worklist;
    for (auto &incoming : incomings)
      worklist.push_back(incoming.pred);
    while (!worklist.empty()) {
      auto *betweenBlock = worklist.pop_back_val();
      if (!betweenSet.insert(betweenBlock).second || betweenBlock == domBlock)
        continue;
      for (auto *pred : betweenBlock->getPredecessors())
        if (domInfo.isReachableFromEntry(pred))
          worklist.push_back(pred);
    }
    SmallVector<Block *> betweenBlocks;
    for (auto *sortedBlock : sortedBlocks)
      if (betweenSet.contains(sortedBlock))
        betweenBlocks.push_back(sortedBlock);

    // A term in the condition under which a value flows into a block argument:
    // either control flowing from `domBlock` to a block, or control flowing
    // into the current block along an incoming edge.
    struct Term {
      Block *block = nullptr;
      Incoming *incoming = nullptr;
    };
    auto getTermDecision = [&](Term term) {
      if (term.incoming)
        return getDecision(*term.incoming);
      return getBranchDecisionsFromDominatorToTarget(builder, domBlock,
                                                     term.block, decisionCache);
    };
    // Estimate the complexity of a term's decision.
    auto getTermWeight = [&](Term term) -> unsigned {
      if (term.incoming)
        return 1;
      auto preds = term.block->getPredecessors();
      return std::max<unsigned>(1, std::distance(preds.begin(), preds.end()));
    };

    // Merge the incoming values of each block argument into a multiplexer
    // chain. The decisions of the incoming edges are mutually exclusive, and
    // exactly one of them holds whenever control reaches the block. This
    // allows us to pick any incoming value as the unconditional base of the
    // chain, and to guard all other values by the disjunction of the decisions
    // of the edges carrying them.
    //
    // To avoid muxes depending on branch decisions that do not affect the
    // argument, we find the blocks from which all reachable incoming edges
    // carry the same value. If control flows through such a uniform block,
    // the argument is guaranteed to have that value if the current block is
    // reached at all. Therefore it is sufficient to check for control flowing
    // into the uniform block, ignoring any branches between it and the current
    // block. Without this, an argument would structurally depend on the
    // conditions of unrelated branches, which may create spurious
    // combinational loops once the surrounding processes are lowered.
    for (auto [argIdx, blockArg] : llvm::enumerate(block->getArguments())) {
      if (incomings.empty())
        break;

      // Determine the blocks from which all reachable incoming edges carry the
      // same value. Visit successors before their predecessors.
      SmallDenseMap<Block *, Value> uniformValues;
      for (auto *betweenBlock : llvm::reverse(betweenBlocks)) {
        Value uniformValue;
        bool isUniform = true;
        auto addValue = [&](Value value) {
          if (!value || (uniformValue && uniformValue != value))
            isUniform = false;
          uniformValue = value;
        };
        if (auto it = incomingIndices.find(betweenBlock);
            it != incomingIndices.end())
          addValue(incomings[it->second].args[argIdx]);
        for (auto *succ : betweenBlock->getSuccessors())
          if (succ != block && betweenSet.contains(succ))
            addValue(uniformValues.lookup(succ));
        if (isUniform)
          uniformValues.insert({betweenBlock, uniformValue});
      }

      // Walk from `domBlock` towards the current block and collect the terms
      // under which each value flows into the argument, stopping at uniform
      // blocks.
      SmallMapVector<Value, SmallVector<Term, 2>, 4> termsByValue;
      SmallPtrSet<Block *, 16> activeBlocks;
      activeBlocks.insert(domBlock);
      for (auto *betweenBlock : betweenBlocks) {
        if (!activeBlocks.contains(betweenBlock))
          continue;
        if (auto value = uniformValues.lookup(betweenBlock)) {
          termsByValue[value].push_back({betweenBlock, nullptr});
          continue;
        }
        if (auto it = incomingIndices.find(betweenBlock);
            it != incomingIndices.end()) {
          auto &incoming = incomings[it->second];
          termsByValue[incoming.args[argIdx]].push_back({nullptr, &incoming});
        }
        for (auto *succ : betweenBlock->getSuccessors())
          if (betweenSet.contains(succ))
            activeBlocks.insert(succ);
      }

      // Pick the value with the most complex condition as the base of the mux
      // chain. On ties, prefer the value carried by the earliest incoming
      // edge, which is the one from the dominating predecessor if there is
      // one.
      SmallDenseMap<Value, unsigned> firstIncomingIndices;
      for (auto [idx, incoming] : llvm::enumerate(incomings))
        firstIncomingIndices.insert({incoming.args[argIdx], idx});
      SmallVector<std::pair<unsigned, unsigned>> weights;
      for (auto &[value, terms] : termsByValue) {
        unsigned weight = 0;
        for (auto term : terms)
          weight += getTermWeight(term);
        weights.push_back({weight, firstIncomingIndices.lookup(value)});
      }
      unsigned baseIdx = 0;
      for (auto [idx, weight] : llvm::enumerate(weights))
        if (weight.first > weights[baseIdx].first ||
            (weight.first == weights[baseIdx].first &&
             weight.second < weights[baseIdx].second))
          baseIdx = idx;

      Value mergedArg = termsByValue.begin()[baseIdx].first;
      for (auto [idx, valueAndTerms] : llvm::enumerate(termsByValue)) {
        if (idx == baseIdx)
          continue;
        auto &[arg, terms] = valueAndTerms;
        auto decision = Condition(false);
        for (auto term : terms)
          decision = decision.orWith(getTermDecision(term), builder);
        if (decision.isTrue())
          mergedArg = arg;
        else if (decision.isFalse())
          continue;
        else
          mergedArg = builder.createOrFold<comb::MuxOp>(
              arg.getLoc(), decision.materialize(builder, arg.getLoc()), arg,
              mergedArg);
      }
      blockArg.replaceAllUsesWith(mergedArg);
    }

    // Move all ops except for the terminator into the entry block.
    if (block != entryBlock)
      entryBlock->getOperations().splice(--entryBlock->end(),
                                         block->getOperations(), block->begin(),
                                         --block->end());
  }

  // Move the yield op into the entry block, replacing the original terminator.
  if (yieldOp != entryBlock->getTerminator()) {
    yieldOp->moveBefore(entryBlock->getTerminator());
    entryBlock->getTerminator()->erase();
  }

  // Remove all blocks except for the entry block. We first clear all operations
  // in the blocks such that the blocks have no more uses in branch ops. Then we
  // remove the blocks themselves in a second pass.
  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->clear();
  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->erase();

  return;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct RemoveControlFlowPass
    : public llhd::impl::RemoveControlFlowPassBase<RemoveControlFlowPass> {
  void runOnOperation() override;
};
} // namespace

void RemoveControlFlowPass::runOnOperation() {
  for (auto op : getOperation().getOps<CombinationalOp>())
    CFRemover(op.getBody()).run();
}
