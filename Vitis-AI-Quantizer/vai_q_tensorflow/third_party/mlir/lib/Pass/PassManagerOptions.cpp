//===- PassManagerOptions.cpp - PassManager Command Line Options ----------===//
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

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

namespace {
struct PassManagerOptions {
  typedef llvm::cl::list<const mlir::PassRegistryEntry *, bool, PassNameParser>
      PassOptionList;

  PassManagerOptions();

  //===--------------------------------------------------------------------===//
  // Multi-threading
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> disableThreads;

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//
  PassOptionList printBefore;
  PassOptionList printAfter;
  llvm::cl::opt<bool> printBeforeAll;
  llvm::cl::opt<bool> printAfterAll;
  llvm::cl::opt<bool> printModuleScope;

  /// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
  void addPrinterInstrumentation(PassManager &pm);

  //===--------------------------------------------------------------------===//
  // Pass Timing
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> passTiming;
  llvm::cl::opt<PassTimingDisplayMode> passTimingDisplayMode;

  /// Add a pass timing instrumentation if enabled by 'pass-timing' flags.
  void addTimingInstrumentation(PassManager &pm);
};
} // end anonymous namespace

static llvm::ManagedStatic<llvm::Optional<PassManagerOptions>> options;

PassManagerOptions::PassManagerOptions()
    //===------------------------------------------------------------------===//
    // Multi-threading
    //===------------------------------------------------------------------===//
    : disableThreads(
          "disable-pass-threading",
          llvm::cl::desc("Disable multithreading in the pass manager"),
          llvm::cl::init(false)),

      //===----------------------------------------------------------------===//
      // IR Printing
      //===----------------------------------------------------------------===//
      printBefore("print-ir-before",
                  llvm::cl::desc("Print IR before specified passes")),
      printAfter("print-ir-after",
                 llvm::cl::desc("Print IR after specified passes")),
      printBeforeAll("print-ir-before-all",
                     llvm::cl::desc("Print IR before each pass"),
                     llvm::cl::init(false)),
      printAfterAll("print-ir-after-all",
                    llvm::cl::desc("Print IR after each pass"),
                    llvm::cl::init(false)),
      printModuleScope(
          "print-ir-module-scope",
          llvm::cl::desc("When printing IR for print-ir-[before|after]{-all} "
                         "always print "
                         "a module IR"),
          llvm::cl::init(false)),

      //===----------------------------------------------------------------===//
      // Pass Timing
      //===----------------------------------------------------------------===//
      passTiming("pass-timing",
                 llvm::cl::desc("Display the execution times of each pass")),
      passTimingDisplayMode(
          "pass-timing-display",
          llvm::cl::desc("Display method for pass timing data"),
          llvm::cl::init(PassTimingDisplayMode::Pipeline),
          llvm::cl::values(
              clEnumValN(PassTimingDisplayMode::List, "list",
                         "display the results in a list sorted by total time"),
              clEnumValN(PassTimingDisplayMode::Pipeline, "pipeline",
                         "display the results with a nested pipeline view"))) {}

/// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
void PassManagerOptions::addPrinterInstrumentation(PassManager &pm) {
  std::function<bool(Pass *)> shouldPrintBeforePass, shouldPrintAfterPass;

  // Handle print-before.
  if (printBeforeAll) {
    // If we are printing before all, then just return true for the filter.
    shouldPrintBeforePass = [](Pass *) { return true; };
  } else if (printBefore.getNumOccurrences() != 0) {
    // Otherwise if there are specific passes to print before, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintBeforePass = [&](Pass *pass) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && llvm::is_contained(printBefore, passInfo);
    };
  }

  // Handle print-after.
  if (printAfterAll) {
    // If we are printing after all, then just return true for the filter.
    shouldPrintAfterPass = [](Pass *) { return true; };
  } else if (printAfter.getNumOccurrences() != 0) {
    // Otherwise if there are specific passes to print after, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintAfterPass = [&](Pass *pass) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && llvm::is_contained(printAfter, passInfo);
    };
  }

  // If there are no valid printing filters, then just return.
  if (!shouldPrintBeforePass && !shouldPrintAfterPass)
    return;

  // Otherwise, add the IR printing instrumentation.
  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      printModuleScope, llvm::errs());
}

/// Add a pass timing instrumentation if enabled by 'pass-timing' flags.
void PassManagerOptions::addTimingInstrumentation(PassManager &pm) {
  if (passTiming)
    pm.enableTiming(passTimingDisplayMode);
}

void mlir::registerPassManagerCLOptions() {
  // Reset the options instance if it hasn't been enabled yet.
  if (!options->hasValue())
    options->emplace();
}

void mlir::applyPassManagerCLOptions(PassManager &pm) {
  // Disable multi-threading.
  if ((*options)->disableThreads)
    pm.disableMultithreading();

  // Add the IR printing instrumentation.
  (*options)->addPrinterInstrumentation(pm);

  // Note: The pass timing instrumentation should be added last to avoid any
  // potential "ghost" timing from other instrumentations being unintentionally
  // included in the timing results.
  (*options)->addTimingInstrumentation(pm);
}
