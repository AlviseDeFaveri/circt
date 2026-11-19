//===- FlattenIO.cpp - HW I/O flattening pass -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_FLATTENIO
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Describes which aggregate types are flattened, and how to split and rebuild
/// values of those types. Structs are always flattened; arrays only if
/// `flattenArrays` is set. Array elements are flattened in index order and
/// named by their index.
struct AggregateFlattener {
  bool flattenArrays = false;

  /// Return the canonical aggregate type if `type` should be flattened, or a
  /// null type otherwise.
  Type getAggregate(Type type) const {
    auto canonical = hw::getCanonicalType(type);
    if (isa<hw::StructType>(canonical))
      return canonical;
    if (auto arrayType = dyn_cast<hw::ArrayType>(canonical))
      if (flattenArrays && arrayType.getNumElements() > 0)
        return canonical;
    return {};
  }

  bool isAggregate(Type type) const { return bool(getAggregate(type)); }

  /// Return the element types of the aggregate `type`, optionally
  /// canonicalized.
  SmallVector<Type> getFieldTypes(Type type, bool canonicalize = false) const {
    SmallVector<Type> types;
    auto maybeCanonical = [&](Type t) {
      return canonicalize ? hw::getCanonicalType(t) : t;
    };
    if (auto structType = dyn_cast<hw::StructType>(getAggregate(type))) {
      for (auto field : structType.getElements())
        types.push_back(maybeCanonical(field.type));
    } else if (auto arrayType = dyn_cast<hw::ArrayType>(getAggregate(type))) {
      types.append(arrayType.getNumElements(),
                   maybeCanonical(arrayType.getElementType()));
    }
    return types;
  }

  /// Return the name suffixes of the elements of the aggregate `type`.
  SmallVector<std::string> getFieldNames(Type type) const {
    SmallVector<std::string> names;
    if (auto structType = dyn_cast<hw::StructType>(getAggregate(type))) {
      for (auto field : structType.getElements())
        names.push_back(field.name.str());
    } else if (auto arrayType = dyn_cast<hw::ArrayType>(getAggregate(type))) {
      for (size_t i = 0, e = arrayType.getNumElements(); i < e; ++i)
        names.push_back(std::to_string(i));
    }
    return names;
  }

  size_t getNumFields(Type type) const { return getFieldTypes(type).size(); }

  /// Split an aggregate value into its elements.
  SmallVector<Value> explode(OpBuilder &builder, Location loc,
                             Value value) const {
    auto type = getAggregate(value.getType());
    if (auto structType = dyn_cast<hw::StructType>(type)) {
      auto explodeOp = hw::StructExplodeOp::create(
          builder, loc, getFieldTypes(structType, /*canonicalize=*/true),
          value);
      return SmallVector<Value>(explodeOp.getResults());
    }
    auto arrayType = cast<hw::ArrayType>(type);
    size_t numElements = arrayType.getNumElements();
    unsigned indexWidth = std::max(1u, llvm::Log2_64_Ceil(numElements));
    auto elementType = hw::getCanonicalType(arrayType.getElementType());
    SmallVector<Value> elements;
    for (size_t i = 0; i < numElements; ++i) {
      auto index = hw::ConstantOp::create(
          builder, loc, builder.getIntegerType(indexWidth), i);
      elements.push_back(
          hw::ArrayGetOp::create(builder, loc, elementType, value, index));
    }
    return elements;
  }

  /// Rebuild a value of aggregate `type` from its elements.
  Value implode(OpBuilder &builder, Location loc, Type type,
                ValueRange elements) const {
    if (isa<hw::StructType>(getAggregate(type)))
      return hw::StructCreateOp::create(builder, loc, type, elements);
    // `hw.array_create` lists the highest index first.
    SmallVector<Value> reversed(llvm::reverse(elements));
    return hw::ArrayCreateOp::create(builder, loc, type, reversed);
  }

  /// Legal if no in- or output type is an aggregate to be flattened.
  bool isLegalModLikeOp(hw::HWModuleLike moduleLikeOp) const {
    return llvm::none_of(moduleLikeOp.getHWModuleType().getPortTypes(),
                         [&](Type type) { return isAggregate(type); });
  }
};

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values)
    llvm::append_range(result, vals);
  return result;
}

// Replaces an output op with a new output with flattened (exploded) structs.
struct OutputOpConversion : public OpConversionPattern<hw::OutputOp> {
  OutputOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     DenseSet<Operation *> *opVisited,
                     const AggregateFlattener &flattener)
      : OpConversionPattern(typeConverter, context), opVisited(opVisited),
        flattener(flattener) {}

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> convOperands;

    // Flatten the operands.
    for (auto operand : adaptor.getOperands()) {
      if (flattener.isAggregate(operand.getType()))
        llvm::append_range(convOperands,
                           flattener.explode(rewriter, op.getLoc(), operand));
      else {
        convOperands.push_back(operand);
      }
    }

    // And replace.
    opVisited->insert(op->getParentOp());
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, convOperands);
    return success();
  }

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> convOperands;

    // Flatten the operands.
    for (auto operand : flattenValues(adaptor.getOperands())) {
      if (flattener.isAggregate(operand.getType()))
        llvm::append_range(convOperands,
                           flattener.explode(rewriter, op.getLoc(), operand));
      else {
        convOperands.push_back(operand);
      }
    }

    // And replace.
    opVisited->insert(op->getParentOp());
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, convOperands);
    return success();
  }
  DenseSet<Operation *> *opVisited;
  const AggregateFlattener &flattener;
};

struct InstanceOpConversion : public OpConversionPattern<hw::InstanceOp> {
  InstanceOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                       DenseSet<hw::InstanceOp> *convertedOps,
                       const StringSet<> *externModules,
                       const AggregateFlattener &flattener)
      : OpConversionPattern(typeConverter, context), convertedOps(convertedOps),
        externModules(externModules), flattener(flattener) {}

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto referencedMod = op.getReferencedModuleNameAttr();
    // If externModules is populated and this is an extern module instance,
    // donot flatten it.
    if (externModules->contains(referencedMod.getValue()))
      return success();

    auto loc = op.getLoc();
    // Flatten the operands.
    llvm::SmallVector<Value> convOperands;
    for (auto operand : flattenValues(adaptor.getOperands())) {
      if (flattener.isAggregate(operand.getType()))
        llvm::append_range(convOperands,
                           flattener.explode(rewriter, loc, operand));
      else {
        convOperands.push_back(operand);
      }
    }

    // Get the new module return type.
    llvm::SmallVector<Type> newResultTypes;
    for (auto oldResultType : op.getResultTypes()) {
      if (flattener.isAggregate(oldResultType))
        llvm::append_range(newResultTypes,
                           flattener.getFieldTypes(oldResultType));
      else
        newResultTypes.push_back(oldResultType);
    }

    // Create the new instance with the flattened module, attributes will be
    // adjusted later.
    auto newInstance = hw::InstanceOp::create(
        rewriter, loc, newResultTypes, op.getInstanceNameAttr(),
        FlatSymbolRefAttr::get(referencedMod), convOperands,
        op.getArgNamesAttr(), op.getResultNamesAttr(), op.getParametersAttr(),
        op.getInnerSymAttr(), op.getDoNotPrintAttr());

    // re-create any structs in the result.
    llvm::SmallVector<Value> convResults;
    size_t oldResultCntr = 0;
    for (size_t resIndex = 0; resIndex < newInstance.getNumResults();
         ++resIndex) {
      Type oldResultType = op.getResultTypes()[oldResultCntr];
      if (auto aggType = flattener.getAggregate(oldResultType)) {
        size_t nElements = flattener.getNumFields(aggType);
        convResults.push_back(flattener.implode(
            rewriter, loc, aggType,
            newInstance.getResults().slice(resIndex, nElements)));
        resIndex += nElements - 1;
      } else
        convResults.push_back(newInstance.getResult(resIndex));

      ++oldResultCntr;
    }
    rewriter.replaceOp(op, convResults);
    convertedOps->insert(newInstance);
    return success();
  }

  DenseSet<hw::InstanceOp> *convertedOps;
  const StringSet<> *externModules;
  const AggregateFlattener &flattener;
};

using IOTypes = std::pair<TypeRange, TypeRange>;

struct IOInfo {
  // A mapping between an arg/res index and the aggregate type of the given
  // field.
  DenseMap<unsigned, Type> argStructs, resStructs;

  // Records of the original arg/res types.
  SmallVector<Type> argTypes, resTypes;
};

class FlattenIOTypeConverter : public TypeConverter {
public:
  FlattenIOTypeConverter(const AggregateFlattener &flattener) {
    addConversion([&flattener](Type type, SmallVectorImpl<Type> &results) {
      if (!flattener.isAggregate(type))
        results.push_back(type);
      else
        llvm::append_range(results, flattener.getFieldTypes(type));
      return success();
    });

    // Materialize !hw.struct<a,b,...> to a, b, ... via. hw.explode. This
    // situation may occur in case of hw.extern_module's with struct outputs.
    addTargetMaterialization(
        [&flattener](OpBuilder &builder, TypeRange resultTypes,
                     ValueRange inputs, Location loc) -> SmallVector<Value> {
          if (inputs.size() != 1 || !flattener.isAggregate(inputs[0].getType()))
            return {};
          return flattener.explode(builder, loc, inputs[0]);
        });
    addTargetMaterialization([&flattener](OpBuilder &builder, Type type,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      if (!flattener.isAggregate(type))
        return {};
      return flattener.implode(builder, loc, type, inputs);
    });

    // In the presence of hw.extern_module which takes struct arguments, we may
    // have materialized struct explodes for said arguments (say, e.g., if the
    // parent module of the hw.instance had structs in its input, and feeds
    // these structs to the hw.instance).
    // These struct explodes needs to be converted back to the original struct,
    // which persist beyond the conversion.
    addSourceMaterialization([&flattener](OpBuilder &builder, Type type,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      if (!flattener.isAggregate(type))
        return {};
      return flattener.implode(builder, loc, type, inputs);
    });
  }
};

} // namespace

template <typename... TOp>
static void addSignatureConversion(DenseMap<Operation *, IOInfo> &ioMap,
                                   ConversionTarget &target,
                                   RewritePatternSet &patterns,
                                   FlattenIOTypeConverter &typeConverter,
                                   const AggregateFlattener &flattener) {
  (hw::populateHWModuleLikeTypeConversionPattern(TOp::getOperationName(),
                                                 patterns, typeConverter),
   ...);

  // Legality is defined by a module having been processed once. This is due to
  // that a pattern cannot be applied multiple times (a 'pattern was already
  // applied' error - a case that would occur for nested structs). Additionally,
  // if a pattern could be applied multiple times, this would complicate
  // updating arg/res names.

  // Instead, we define legality as when a module has had a modification to its
  // top-level i/o. This ensures that only a single level of structs are
  // processed during signature conversion, which then allows us to use the
  // signature conversion in a recursive manner.
  target.addDynamicallyLegalOp<TOp...>([&](hw::HWModuleLike moduleLikeOp) {
    if (flattener.isLegalModLikeOp(moduleLikeOp))
      return true;

    // This op is involved in conversion. Check if the signature has changed.
    auto ioInfoIt = ioMap.find(moduleLikeOp);
    if (ioInfoIt == ioMap.end()) {
      // Op wasn't primed in the map. Do the safe thing, assume
      // that it's not considered in this pass, and mark it as legal
      return true;
    }
    auto ioInfo = ioInfoIt->second;

    auto compareTypes = [&](TypeRange oldTypes, TypeRange newTypes) {
      return llvm::any_of(llvm::zip(oldTypes, newTypes), [&](auto typePair) {
        auto oldType = std::get<0>(typePair);
        auto newType = std::get<1>(typePair);
        return oldType != newType;
      });
    };
    auto mtype = moduleLikeOp.getHWModuleType();
    if (compareTypes(mtype.getOutputTypes(), ioInfo.resTypes) ||
        compareTypes(mtype.getInputTypes(), ioInfo.argTypes))
      return true;

    // We're pre-conversion for an op that was primed in the map - it will
    // always be illegal since it has to-be-converted struct types at its I/O.
    return false;
  });
}

template <typename T>
static bool hasUnconvertedOps(mlir::ModuleOp module,
                              const AggregateFlattener &flattener) {
  return llvm::any_of(module.getBody()->getOps<T>(),
                      [&](T op) { return !flattener.isLegalModLikeOp(op); });
}

template <typename T>
static DenseMap<Operation *, IOTypes> populateIOMap(mlir::ModuleOp module) {
  DenseMap<Operation *, IOTypes> ioMap;
  for (auto op : module.getOps<T>())
    ioMap[op] = {op.getArgumentTypes(), op.getResultTypes()};
  return ioMap;
}

template <typename ModTy, typename T>
static llvm::SmallVector<Attribute>
updateNameAttribute(ModTy op, StringRef attrName,
                    DenseMap<unsigned, Type> &structMap, T oldNames,
                    char joinChar, const AggregateFlattener &flattener) {
  llvm::SmallVector<Attribute> newNames;
  for (auto [i, oldName] : llvm::enumerate(oldNames)) {
    // Was this arg/res index a struct?
    auto it = structMap.find(i);
    if (it == structMap.end()) {
      // No, keep old name.
      newNames.push_back(StringAttr::get(op->getContext(), oldName));
      continue;
    }

    // Yes - create new names from the struct fields and the old name at the
    // index.
    for (auto &field : flattener.getFieldNames(it->second))
      newNames.push_back(
          StringAttr::get(op->getContext(), oldName + Twine(joinChar) + field));
  }
  return newNames;
}

template <typename ModTy>
static void updateModulePortNames(ModTy op, hw::ModuleType oldModType,
                                  char joinChar,
                                  const AggregateFlattener &flattener) {
  // Module arg and result port names may not be ordered. So we cannot reuse
  // updateNameAttribute. The arg and result order must be preserved.
  SmallVector<Attribute> newNames;
  SmallVector<hw::ModulePort> oldPorts(oldModType.getPorts().begin(),
                                       oldModType.getPorts().end());
  for (auto oldPort : oldPorts) {
    auto oldName = oldPort.name;
    if (flattener.isAggregate(oldPort.type)) {
      for (auto &field : flattener.getFieldNames(oldPort.type))
        newNames.push_back(StringAttr::get(
            op->getContext(), oldName.getValue() + Twine(joinChar) + field));
    } else
      newNames.push_back(oldName);
  }
  op.setAllPortNames(newNames);
}

static llvm::SmallVector<Location>
updateLocAttribute(DenseMap<unsigned, Type> &structMap,
                   SmallVectorImpl<Location> &oldLocs,
                   const AggregateFlattener &flattener) {
  llvm::SmallVector<Location> newLocs;
  for (auto [i, oldLoc] : llvm::enumerate(oldLocs)) {
    // Was this arg/res index a struct?
    auto it = structMap.find(i);
    if (it == structMap.end()) {
      // No, keep old name.
      newLocs.push_back(oldLoc);
      continue;
    }

    newLocs.append(flattener.getNumFields(it->second), oldLoc);
  }
  return newLocs;
}

/// The conversion framework seems to throw away block argument locations.  We
/// use this function to copy the location from the original argument to the
/// set of flattened arguments.
static void updateBlockLocations(hw::HWModuleLike op,
                                 DenseMap<unsigned, Type> &structMap) {
  auto locs = op.getInputLocs();
  if (locs.empty() || op.getModuleBody().empty())
    return;
  for (auto [arg, loc] : llvm::zip(op.getBodyBlock()->getArguments(), locs))
    arg.setLoc(loc);
}

static void setIOInfo(hw::HWModuleLike op, IOInfo &ioInfo,
                      const AggregateFlattener &flattener) {
  ioInfo.argTypes = op.getInputTypes();
  ioInfo.resTypes = op.getOutputTypes();
  for (auto [i, arg] : llvm::enumerate(ioInfo.argTypes)) {
    if (auto aggType = flattener.getAggregate(arg))
      ioInfo.argStructs[i] = aggType;
  }
  for (auto [i, res] : llvm::enumerate(ioInfo.resTypes)) {
    if (auto aggType = flattener.getAggregate(res))
      ioInfo.resStructs[i] = aggType;
  }
}

template <typename T>
static DenseMap<Operation *, IOInfo>
populateIOInfoMap(mlir::ModuleOp module, const AggregateFlattener &flattener) {
  DenseMap<Operation *, IOInfo> ioInfoMap;
  for (auto op : module.getOps<T>()) {
    IOInfo ioInfo;
    setIOInfo(op, ioInfo, flattener);
    ioInfoMap[op] = ioInfo;
  }
  return ioInfoMap;
}

template <typename T>
static LogicalResult flattenOpsOfType(ModuleOp module, bool recursive,
                                      StringSet<> &externModules, char joinChar,
                                      const AggregateFlattener &flattener) {
  auto *ctx = module.getContext();
  FlattenIOTypeConverter typeConverter(flattener);

  // Recursively (in case of nested structs) lower the module. We do this one
  // conversion at a time to allow for updating the arg/res names of the
  // module in between flattening each level of structs.
  while (hasUnconvertedOps<T>(module, flattener)) {
    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    target.addLegalDialect<hw::HWDialect>();

    // Record any struct types at the module signature. This will be used
    // post-conversion to update the argument and result names.
    auto ioInfoMap = populateIOInfoMap<T>(module, flattener);

    // Record the instances that were converted. We keep these around since we
    // need to update their arg/res attribute names after the modules themselves
    // have been updated.
    llvm::DenseSet<hw::InstanceOp> convertedInstances;

    // Argument conversion for output ops. Similarly to the signature
    // conversion, legality is based on the op having been visited once, due to
    // the possibility of nested structs.
    DenseSet<Operation *> opVisited;
    patterns.add<OutputOpConversion>(typeConverter, ctx, &opVisited, flattener);

    patterns.add<InstanceOpConversion>(typeConverter, ctx, &convertedInstances,
                                       &externModules, flattener);
    target.addDynamicallyLegalOp<hw::OutputOp>(
        [&](auto op) { return opVisited.contains(op->getParentOp()); });
    target.addDynamicallyLegalOp<hw::InstanceOp>([&](hw::InstanceOp op) {
      auto refName = op.getReferencedModuleName();
      return externModules.contains(refName) ||
             (llvm::none_of(op->getOperands(),
                            [&](auto operand) {
                              return flattener.isAggregate(operand.getType());
                            }) &&
              llvm::none_of(op->getResultTypes(), [&](auto result) {
                return flattener.isAggregate(result);
              }));
    });

    DenseMap<Operation *, ArrayAttr> oldArgNames, oldResNames;
    DenseMap<Operation *, SmallVector<Location>> oldArgLocs, oldResLocs;
    DenseMap<Operation *, hw::ModuleType> oldModTypes;

    for (auto op : module.getOps<T>()) {
      oldModTypes[op] = op.getHWModuleType();
      oldArgNames[op] = ArrayAttr::get(module.getContext(), op.getInputNames());
      oldResNames[op] =
          ArrayAttr::get(module.getContext(), op.getOutputNames());
      oldArgLocs[op] = op.getInputLocs();
      oldResLocs[op] = op.getOutputLocs();
    }

    // Signature conversion and legalization patterns.
    addSignatureConversion<T>(ioInfoMap, target, patterns, typeConverter,
                              flattener);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return failure();

    // Update the arg/res names of the module.
    for (auto op : module.getOps<T>()) {
      auto ioInfo = ioInfoMap[op];
      updateModulePortNames(op, oldModTypes[op], joinChar, flattener);
      auto newArgLocs =
          updateLocAttribute(ioInfo.argStructs, oldArgLocs[op], flattener);
      auto newResLocs =
          updateLocAttribute(ioInfo.resStructs, oldResLocs[op], flattener);
      newArgLocs.append(newResLocs.begin(), newResLocs.end());
      op.setAllPortLocs(newArgLocs);
      updateBlockLocations(op, ioInfo.argStructs);
    }

    // And likewise with the converted instance ops.
    for (auto instanceOp : convertedInstances) {
      auto targetModule =
          cast<hw::HWModuleLike>(SymbolTable::lookupNearestSymbolFrom(
              instanceOp, instanceOp.getReferencedModuleNameAttr()));

      IOInfo ioInfo;
      if (!ioInfoMap.contains(targetModule)) {
        // If an extern module, then not yet processed, populate the maps.
        setIOInfo(targetModule, ioInfo, flattener);
        ioInfoMap[targetModule] = ioInfo;
        oldArgNames[targetModule] =
            ArrayAttr::get(module.getContext(), targetModule.getInputNames());
        oldResNames[targetModule] =
            ArrayAttr::get(module.getContext(), targetModule.getOutputNames());
        oldArgLocs[targetModule] = targetModule.getInputLocs();
        oldResLocs[targetModule] = targetModule.getOutputLocs();
      } else
        ioInfo = ioInfoMap[targetModule];

      instanceOp.setInputNames(ArrayAttr::get(
          instanceOp.getContext(),
          updateNameAttribute(
              instanceOp, "argNames", ioInfo.argStructs,
              oldArgNames[targetModule].template getAsValueRange<StringAttr>(),
              joinChar, flattener)));
      instanceOp.setOutputNames(ArrayAttr::get(
          instanceOp.getContext(),
          updateNameAttribute(
              instanceOp, "resultNames", ioInfo.resStructs,
              oldResNames[targetModule].template getAsValueRange<StringAttr>(),
              joinChar, flattener)));
    }

    // Break if we've only lowering a single level of structs.
    if (!recursive)
      break;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

template <typename... TOps>
static bool flattenIO(ModuleOp module, bool recursive,
                      StringSet<> &externModules, char joinChar,
                      const AggregateFlattener &flattener) {
  return (failed(flattenOpsOfType<TOps>(module, recursive, externModules,
                                        joinChar, flattener)) ||
          ...);
}

namespace {

class FlattenIOPass : public circt::hw::impl::FlattenIOBase<FlattenIOPass> {
  using Base::Base;

public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    AggregateFlattener flattener{flattenArrays};
    if (!flattenExtern) {
      // Record the extern modules, do not flatten them.
      for (auto m : module.getOps<hw::HWModuleExternOp>())
        externModules.insert(m.getModuleName());
      if (flattenIO<hw::HWModuleOp, hw::HWModuleGeneratedOp>(
              module, recursive, externModules, joinChar, flattener))
        signalPassFailure();
      return;
    }

    if (flattenIO<hw::HWModuleOp, hw::HWModuleExternOp,
                  hw::HWModuleGeneratedOp>(module, recursive, externModules,
                                           joinChar, flattener))
      signalPassFailure();
  };

private:
  StringSet<> externModules;
};
} // namespace
