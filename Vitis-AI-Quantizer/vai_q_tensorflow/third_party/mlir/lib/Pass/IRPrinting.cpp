//===- IRPrinting.cpp -----------------------------------------------------===//
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

#include "PassDetail.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
class IRPrinterInstrumentation : public PassInstrumentation {
public:
  /// A filter function to decide if the given pass should be printed. Returns
  /// true if the pass should be printed, false otherwise.
  using ShouldPrintFn = std::function<bool(Pass *)>;

  IRPrinterInstrumentation(ShouldPrintFn &&shouldPrintBeforePass,
                           ShouldPrintFn &&shouldPrintAfterPass,
                           bool printModuleScope, raw_ostream &out)
      : shouldPrintBeforePass(shouldPrintBeforePass),
        shouldPrintAfterPass(shouldPrintAfterPass),
        printModuleScope(printModuleScope), out(out) {
    assert((shouldPrintBeforePass || shouldPrintAfterPass) &&
           "expected atleast one valid filter function");
  }

private:
  /// Instrumentation hooks.
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

  /// Filter functions for before and after pass execution.
  ShouldPrintFn shouldPrintBeforePass, shouldPrintAfterPass;

  /// Flag to toggle if the printer should always print at module scope.
  bool printModuleScope;

  /// The stream to output to.
  raw_ostream &out;
};
} // end anonymous namespace

/// Returns true if the given pass is hidden from IR printing.
static bool isHiddenPass(Pass *pass) {
  return isAdaptorPass(pass) || isVerifierPass(pass);
}

static void printIR(Operation *op, bool printModuleScope, raw_ostream &out) {
  // Check for printing at module scope.
  auto function = dyn_cast<FuncOp>(op);
  if (printModuleScope && function) {
    // Print the function name and a newline before the Module.
    out << " (function: " << function.getName() << ")\n";
    function.getParentOfType<ModuleOp>().print(out);
    return;
  }

  // Print a newline before the IR.
  out << "\n";

  // Print the given function.
  if (function) {
    function.print(out);
    return;
  }

  // Print the given module.
  assert(isa<ModuleOp>(op) && "unexpected IR unit");
  cast<ModuleOp>(op).print(out);
}

/// Instrumentation hooks.
void IRPrinterInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  // Skip hidden passes and passes that the user filtered out.
  if (!shouldPrintBeforePass || isHiddenPass(pass) ||
      !shouldPrintBeforePass(pass))
    return;
  out << formatv("*** IR Dump Before {0} ***", pass->getName());
  printIR(op, printModuleScope, out);
  out << "\n\n";
}

void IRPrinterInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  // Skip hidden passes and passes that the user filtered out.
  if (!shouldPrintAfterPass || isHiddenPass(pass) ||
      !shouldPrintAfterPass(pass))
    return;
  out << formatv("*** IR Dump After {0} ***", pass->getName());
  printIR(op, printModuleScope, out);
  out << "\n\n";
}

void IRPrinterInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  // Skip adaptor passes and passes that the user filtered out.
  if (!shouldPrintAfterPass || isAdaptorPass(pass) ||
      !shouldPrintAfterPass(pass))
    return;
  out << formatv("*** IR Dump After {0} Failed ***", pass->getName());
  printIR(op, printModuleScope, out);
  out << "\n\n";
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

/// Add an instrumentation to print the IR before and after pass execution.
void PassManager::enableIRPrinting(
    std::function<bool(Pass *)> shouldPrintBeforePass,
    std::function<bool(Pass *)> shouldPrintAfterPass, bool printModuleScope,
    raw_ostream &out) {
  addInstrumentation(new IRPrinterInstrumentation(
      std::move(shouldPrintBeforePass), std::move(shouldPrintAfterPass),
      printModuleScope, out));
}
