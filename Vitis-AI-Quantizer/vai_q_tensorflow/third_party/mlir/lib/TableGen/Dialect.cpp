//===- Dialect.cpp - Dialect wrapper class --------------------------------===//
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
// Dialect wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Dialect.h"
#include "llvm/TableGen/Record.h"

namespace mlir {
namespace tblgen {

StringRef tblgen::Dialect::getName() const {
  return def.getValueAsString("name");
}

StringRef tblgen::Dialect::getCppNamespace() const {
  return def.getValueAsString("cppNamespace");
}

} // end namespace tblgen
} // end namespace mlir
