//===- NestedMacher.h - Nested matcher for MLFunction -----------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_MLFUNCTIONMATCHER_H_
#define MLIR_ANALYSIS_MLFUNCTIONMATCHER_H_

#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Allocator.h"

namespace mlir {

struct NestedPattern;
class Operation;

/// An NestedPattern captures nested patterns in the IR.
/// It is used in conjunction with a scoped NestedPatternContext which is an
/// llvm::BumpPtrAllocator that handles memory allocations efficiently and
/// avoids ownership issues.
///
/// In order to use NestedPatterns, first create a scoped context.
/// When the context goes out of scope, everything is freed.
/// This design simplifies the API by avoiding references to the context and
/// makes it clear that references to matchers must not escape.
///
/// Example:
///   {
///      NestedPatternContext context;
///      auto gemmLike = Doall(Doall(Red(LoadStores())));
///      auto matches = gemmLike.match(f);
///      // do work on matches
///   }  // everything is freed
///
///
/// Nested abstraction for matching results.
/// Provides access to the nested Operation* captured by a Matcher.
///
/// A NestedMatch contains an Operation* and the children NestedMatch and is
/// thus cheap to copy. NestedMatch is stored in a scoped bumper allocator whose
/// lifetime is managed by an RAII NestedPatternContext.
struct NestedMatch {
  static NestedMatch build(Operation *operation,
                           ArrayRef<NestedMatch> nestedMatches);
  NestedMatch(const NestedMatch &) = default;
  NestedMatch &operator=(const NestedMatch &) = default;

  explicit operator bool() { return matchedOperation != nullptr; }

  Operation *getMatchedOperation() { return matchedOperation; }
  ArrayRef<NestedMatch> getMatchedChildren() { return matchedChildren; }

private:
  friend struct NestedPattern;
  friend struct NestedPatternContext;

  /// Underlying global bump allocator managed by a NestedPatternContext.
  static llvm::BumpPtrAllocator *&allocator();

  NestedMatch() = default;

  /// Payload, holds a NestedMatch and all its children along this branch.
  Operation *matchedOperation;
  ArrayRef<NestedMatch> matchedChildren;
};

/// A NestedPattern is a nested operation walker that:
///   1. recursively matches a substructure in the tree;
///   2. uses a filter function to refine matches with extra semantic
///      constraints (passed via a lambda of type FilterFunctionType);
///   3. TODO(ntv) optionally applies actions (lambda).
///
/// Nested patterns are meant to capture imperfectly nested loops while matching
/// properties over the whole loop nest. For instance, in vectorization we are
/// interested in capturing all the imperfectly nested loops of a certain type
/// and such that all the load and stores have certain access patterns along the
/// loops' induction variables). Such NestedMatches are first captured using the
/// `match` function and are later processed to analyze properties and apply
/// transformations in a non-greedy way.
///
/// The NestedMatches captured in the IR can grow large, especially after
/// aggressive unrolling. As experience has shown, it is generally better to use
/// a plain walk over operations to match flat patterns but the current
/// implementation is competitive nonetheless.
using FilterFunctionType = std::function<bool(Operation &)>;
inline bool defaultFilterFunction(Operation &) { return true; }
struct NestedPattern {
  NestedPattern(ArrayRef<NestedPattern> nested,
                FilterFunctionType filter = defaultFilterFunction);
  NestedPattern(const NestedPattern &) = default;
  NestedPattern &operator=(const NestedPattern &) = default;

  /// Returns all the top-level matches in `func`.
  void match(FuncOp func, SmallVectorImpl<NestedMatch> *matches) {
    func.walk([&](Operation *op) { matchOne(op, matches); });
  }

  /// Returns all the top-level matches in `op`.
  void match(Operation *op, SmallVectorImpl<NestedMatch> *matches) {
    op->walk([&](Operation *child) { matchOne(child, matches); });
  }

  /// Returns the depth of the pattern.
  unsigned getDepth() const;

private:
  friend struct NestedPatternContext;
  friend struct NestedMatch;
  friend struct State;

  /// Underlying global bump allocator managed by a NestedPatternContext.
  static llvm::BumpPtrAllocator *&allocator();

  /// Matches this pattern against a single `op` and fills matches with the
  /// result.
  void matchOne(Operation *op, SmallVectorImpl<NestedMatch> *matches);

  /// Nested patterns to be matched.
  ArrayRef<NestedPattern> nestedPatterns;

  /// Extra filter function to apply to prune patterns as the IR is walked.
  FilterFunctionType filter;

  /// skip is an implementation detail needed so that we can implement match
  /// without switching on the type of the Operation. The idea is that a
  /// NestedPattern first checks if it matches locally and then recursively
  /// applies its nested matchers to its elem->nested. Since we want to rely on
  /// the existing operation walking functionality rather than duplicate
  /// it, we allow an off-by-one traversal to account for the fact that we
  /// write:
  ///
  ///  void match(Operation *elem) {
  ///    for (auto &c : getNestedPatterns()) {
  ///      NestedPattern childPattern(...);
  ///                                  ^~~~ Needs off-by-one skip.
  ///
  Operation *skip;
};

/// RAII structure to transparently manage the bump allocator for
/// NestedPattern and NestedMatch classes. This avoids passing a context to
/// all the API functions.
struct NestedPatternContext {
  NestedPatternContext() {
    assert(NestedMatch::allocator() == nullptr &&
           "Only a single NestedPatternContext is supported");
    assert(NestedPattern::allocator() == nullptr &&
           "Only a single NestedPatternContext is supported");
    NestedMatch::allocator() = &allocator;
    NestedPattern::allocator() = &allocator;
  }
  ~NestedPatternContext() {
    NestedMatch::allocator() = nullptr;
    NestedPattern::allocator() = nullptr;
  }
  llvm::BumpPtrAllocator allocator;
};

namespace matcher {
// Syntactic sugar NestedPattern builder functions.
NestedPattern Op(FilterFunctionType filter = defaultFilterFunction);
NestedPattern If(NestedPattern child);
NestedPattern If(FilterFunctionType filter, NestedPattern child);
NestedPattern If(ArrayRef<NestedPattern> nested = {});
NestedPattern If(FilterFunctionType filter,
                 ArrayRef<NestedPattern> nested = {});
NestedPattern For(NestedPattern child);
NestedPattern For(FilterFunctionType filter, NestedPattern child);
NestedPattern For(ArrayRef<NestedPattern> nested = {});
NestedPattern For(FilterFunctionType filter,
                  ArrayRef<NestedPattern> nested = {});

bool isParallelLoop(Operation &op);
bool isReductionLoop(Operation &op);
bool isLoadOrStore(Operation &op);

} // end namespace matcher
} // end namespace mlir

#endif // MLIR_ANALYSIS_MLFUNCTIONMATCHER_H_
