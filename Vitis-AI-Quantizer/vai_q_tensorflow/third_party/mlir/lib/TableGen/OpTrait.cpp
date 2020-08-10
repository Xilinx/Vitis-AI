//===- OpTrait.cpp - OpTrait class ----------------------------------------===//
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
//
// OpTrait wrapper to simplify using TableGen Record defining a MLIR OpTrait.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

mlir::tblgen::OpTrait mlir::tblgen::OpTrait::create(const llvm::Init *init) {
  auto def = cast<llvm::DefInit>(init)->getDef();
  if (def->isSubClassOf("PredOpTrait"))
    return OpTrait(Kind::Pred, def);
  if (def->isSubClassOf("GenInternalOpTrait"))
    return OpTrait(Kind::Internal, def);
  assert(def->isSubClassOf("NativeOpTrait"));
  return OpTrait(Kind::Native, def);
}

mlir::tblgen::OpTrait::OpTrait(Kind kind, const llvm::Record *def)
    : def(def), kind(kind) {}

llvm::StringRef mlir::tblgen::NativeOpTrait::getTrait() const {
  return def->getValueAsString("trait");
}

llvm::StringRef mlir::tblgen::InternalOpTrait::getTrait() const {
  return def->getValueAsString("trait");
}

std::string mlir::tblgen::PredOpTrait::getPredTemplate() const {
  auto pred = tblgen::Pred(def->getValueInit("predicate"));
  return pred.getCondition();
}

llvm::StringRef mlir::tblgen::PredOpTrait::getDescription() const {
  return def->getValueAsString("description");
}
