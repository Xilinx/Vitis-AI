//===- ReferenceImplGen.cpp - MLIR reference implementation generator -----===//
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
// ReferenceImplGen uses the description of operations to generate reference
// implementations for the ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

using mlir::tblgen::Operator;

static void emitReferenceImplementations(const RecordKeeper &recordKeeper,
                                         raw_ostream &os) {
  emitSourceFileHeader("Reference implementation file", os);
  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");

  os << "void printRefImplementation(StringRef opName, mlir::FuncOp *f) {\n"
     << "  using namespace ::mlir::edsc;\n"
     << "if (false) {}";
  for (auto *def : defs) {
    Operator op(def);
    auto referenceImplGenerator = def->getValueInit("referenceImplementation");
    if (!referenceImplGenerator)
      continue;
    os << " else if (opName == \"" << op.getOperationName() << "\") {\n"
       << "  edsc::ScopedContext scope(f);\n";

    for (auto en : llvm::enumerate(op.getOperands())) {
      os.indent(2) << formatv("ValueHandle arg_{0}(f->getArgument({1})); "
                              "(void)arg_{0};\n",
                              en.value().name, en.index());
      // TODO(jpienaar): this is generally incorrect, not all args are memref
      // in the general case.
      os.indent(2) << formatv("MemRefView view_{0}(f->getArgument({1})); "
                              "(void)view_{0};\n",
                              en.value().name, en.index());
    }
    unsigned numOperands = op.getNumOperands();
    unsigned numResults = op.getNumResults();
    for (unsigned idx = 0; idx < numResults; ++idx) {
      os.indent(2) << formatv("ValueHandle arg_{0}(f->getArgument({1})); "
                              "(void)arg_{0};\n",
                              op.getResult(idx).name, numOperands + idx);
      // TODO(jpienaar): this is generally incorrect, not all args are memref
      // in the general case.
      os.indent(2) << formatv("MemRefView view_{0}(f->getArgument({1})); "
                              "(void)view_{0};\n",
                              op.getResult(idx).name, numOperands + idx);
    }

    // Print the EDSC.
    os << referenceImplGenerator->getAsUnquotedString() << "\n";
    os.indent(2) << "f->print(llvm::outs());\n\n";
    os << "}";
  }
  os << " else {\n";
  os.indent(2) << "f->emitError(\"no reference impl. for \" + opName);\n";
  os.indent(2) << "return;\n";
  os << "}\n";
  os << "}\n";
}

static mlir::GenRegistration
    genRegister("gen-reference-implementations",
                "Generate reference implemenations",
                [](const RecordKeeper &records, raw_ostream &os) {
                  emitReferenceImplementations(records, os);
                  return false;
                });
