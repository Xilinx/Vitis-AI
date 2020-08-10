//===- UniformConstraints.cpp - Constraints for uniform quant -------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Quantizer/Support/UniformConstraints.h"

#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Quantizer/Support/Configuration.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"
#include "mlir/Quantizer/Support/Metadata.h"
#include "mlir/Quantizer/Support/Rules.h"
#include "mlir/Quantizer/Support/TypeUtils.h"
#include "mlir/Quantizer/Support/UniformSolvers.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::quantizer;
using namespace mlir::quant;

namespace {

struct ClusteredFacts {
  ExpandingMinMaxFact requiredRange;
  DiscreteScaleZeroPointFact explicitScaleZeroPoint;
};

} // end anonymous namespace

static QuantizedType solveUniformType(SolverContext &solverContext,
                                      const ClusteredFacts &clusteredFacts,
                                      const CandidateQuantizedType &ct,
                                      Type originalElementType, Location loc) {
  switch (ct.scheme) {
  default:
    emitError(loc, "unsupported scheme for uniform type conversion");
    return nullptr;

  case CandidateQuantizedType::Scheme::UniformPerLayer: {
    if (!clusteredFacts.requiredRange.hasValue()) {
      // TODO: Issue some kind of diagnostic. This is not an error.
      return nullptr;
    }

    uint64_t numLevels = ct.quantizedType.getStorageTypeMax() -
                         ct.quantizedType.getStorageTypeMin();
    UniformStorageParams params{numLevels,
                                ct.quantizedType.getStorageTypeMin()};
    UniformParamsFromMinMaxSolver solver(
        params, clusteredFacts.requiredRange.getValue().first,
        clusteredFacts.requiredRange.getValue().second);
    if (!solver.compute()) {
      emitWarning(loc) << "unable to solve uniform type with "
                       << "UniformParamsFromMinMaxSolver";
      return nullptr;
    }

    return UniformQuantizedType::getChecked(
        ct.quantizedType.getFlags(), ct.quantizedType.getStorageType(),
        originalElementType, solver.getScale(), solver.getZp(),
        ct.quantizedType.getStorageTypeMin(),
        ct.quantizedType.getStorageTypeMax(), loc);
  }
  case CandidateQuantizedType::Scheme::UniformExplicitFixedPointScale: {
    if (!clusteredFacts.explicitScaleZeroPoint.hasValue()) {
      emitRemark(loc)
          << "unable to solve uniform type with UniformExplicitFixedPointScale "
          << "(no explicitScaleZeroPoint)";
      return nullptr;
    }

    const auto &scaleZp = clusteredFacts.explicitScaleZeroPoint.getValue();
    assert(scaleZp.value && "optional value not set on fact");

    if (scaleZp.conflict) {
      emitWarning(loc)
          << "conflicting explicit scale/zeroPoint on node cluster: "
          << "an arbitrary scale/zeroPoint will be used";
    }

    return UniformQuantizedType::getChecked(
        ct.quantizedType.getFlags(), ct.quantizedType.getStorageType(),
        originalElementType,
        scaleZp.value->first, // scale
        0, // zeroPoint (fixed point solutions only for this scheme)
        ct.quantizedType.getStorageTypeMin(),
        ct.quantizedType.getStorageTypeMax(), loc);

    return nullptr;
  }
  }
}

namespace {

class PropagateExplicitScale : public CAGConstraintNode {
public:
  PropagateExplicitScale()
      : CAGConstraintNode(Kind::UniformPropagateExplicitScale) {}
  static bool classof(const CAGNode *n) {
    return n->getKind() == Kind::Constraint ||
           n->getKind() == Kind::UniformPropagateExplicitScale;
  }

private:
  void printLabel(llvm::raw_ostream &os) const override {
    os << "PropagateExplicitScale";
  }
  void propagate(SolverContext &solverContext,
                 const TargetConfiguration &config) override {
    DiscreteScaleZeroPointFact scaleZp;

    // Get scale/zp from all parents.
    for (auto it = incoming_begin(), e = incoming_end(); it != e; ++it) {
      auto parentAnchor = llvm::cast<CAGAnchorNode>(*it);
      auto selectedType = parentAnchor->getUniformMetadata().selectedType;
      if (auto uqType = selectedType.dyn_cast_or_null<UniformQuantizedType>()) {
        scaleZp.assertValue(
            CAGUniformMetadata::SalienceRequired,
            std::make_pair(uqType.getScale(), static_cast<int64_t>(0)));
      }
    }

    // Propagate to children.
    if (scaleZp.hasValue()) {
      for (auto it = begin(), e = end(); it != e; ++it) {
        auto childAnchor = llvm::cast<CAGAnchorNode>(*it);
        if (modified(childAnchor->getUniformMetadata()
                         .explicitScaleZeroPoint.mergeFrom(scaleZp))) {
          childAnchor->markDirty();
        }
      }
    }
  }
};

/// A constraint node which will solve uniform quantization for all parents
/// of the constraint, assuming that they are coupled.
class SolveUniformConstraintNode : public CAGConstraintNode {
public:
  SolveUniformConstraintNode()
      : CAGConstraintNode(Kind::SolveUniformConstraint) {
    markDirty();
  }
  static bool classof(const CAGNode *n) {
    return n->getKind() == Kind::Constraint ||
           n->getKind() == Kind::SolveUniformConstraint;
  }

private:
  void printLabel(llvm::raw_ostream &os) const override {
    os << "SolveUniform";
  }

  void propagate(SolverContext &solverContext,
                 const TargetConfiguration &config) override {
    // First determine the required min/max range and type constraints.
    Location fusedLoc = UnknownLoc::get(&solverContext.getMlirContext());
    llvm::SmallBitVector enabledCandidateTypesMask(
        config.getAllCandidateTypesMask());
    ClusteredFacts clusteredFacts;
    Type originalElementType;
    for (auto it = incoming_begin(), e = incoming_end(); it != e; ++it) {
      auto parentAnchor = llvm::cast<CAGAnchorNode>(*it);
      auto metadata = parentAnchor->getUniformMetadata();
      // TODO: Possibly use a location that fuses all involved parents.
      fusedLoc = parentAnchor->getOp()->getLoc();

      // Shared element type.
      auto parentOriginalElementType =
          getElementOrPrimitiveType(parentAnchor->getOriginalType());
      if (!originalElementType) {
        originalElementType = parentOriginalElementType;
      } else {
        if (originalElementType != parentOriginalElementType) {
          parentAnchor->getOp()->emitError()
              << "cannot compute uniform type: parent element types mismatch";
          return;
        }
      }
      // Range.
      clusteredFacts.requiredRange.mergeFrom(metadata.requiredRange);

      // Explicit scale and zero point.
      clusteredFacts.explicitScaleZeroPoint.mergeFrom(
          metadata.explicitScaleZeroPoint);

      // Shared candidate types.
      enabledCandidateTypesMask.reset(metadata.disabledCandidateTypes);
    }

    // Find the first enabled candidate type.
    const CandidateQuantizedType *bestCandidateType = nullptr;
    for (auto &ct : config.getCandidateTypes()) {
      if (enabledCandidateTypesMask.test(ct.ordinal)) {
        bestCandidateType = &ct;
        break;
      }
    }

    if (!bestCandidateType || !originalElementType) {
      emitRemark(fusedLoc)
          << "not solving uniform type (no viable candidate type)";
      return;
    }

    // Solve for the type.
    QuantizedType selectedType =
        solveUniformType(solverContext, clusteredFacts, *bestCandidateType,
                         originalElementType, fusedLoc);

    // Apply it to all parents.
    for (auto it = incoming_begin(), e = incoming_end(); it != e; ++it) {
      auto parentAnchor = llvm::cast<CAGAnchorNode>(*it);
      auto &metadata = parentAnchor->getUniformMetadata();
      if (metadata.selectedType != selectedType) {
        metadata.selectedType = selectedType;
        // And mark all children of the parent dirty (except us).
        for (auto child : *parentAnchor) {
          if (child != this) {
            child->markDirty();
          }
        }
      }
    }
  }
};

} // end anonymous namespace

void UniformConstraintsBuilder::coupleAnchors(CAGAnchorNode *a,
                                              CAGAnchorNode *b) {
  slice.addClusteredConstraint<SolveUniformConstraintNode>({a, b});
}

void UniformConstraintsBuilder::applyStats(CAGAnchorNode *a,
                                           TensorAxisStatistics stats) {
  a->getUniformMetadata().requiredRange.assertValue(
      CAGUniformMetadata::SalienceDefault, {stats.minValue, stats.maxValue});
}

void UniformConstraintsBuilder::clamp(CAGAnchorNode *a, APFloat minValue,
                                      APFloat maxValue) {
  a->getUniformMetadata().requiredRange.assertValue(
      CAGUniformMetadata::SalienceDefault,
      {minValue.convertToDouble(), maxValue.convertToDouble()});
}

void UniformConstraintsBuilder::propagateExplicitScale(CAGAnchorNode *from,
                                                       CAGAnchorNode *to) {
  slice.addUnidirectionalConstraint<PropagateExplicitScale>(from, {to});
}
