//===- Parser.h - MLIR Parser Library Interface -----------------*- C++ -*-===//
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
// This file is contains the interface to the MLIR parser library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PARSER_H
#define MLIR_PARSER_H

namespace llvm {
class SourceMgr;
class SMDiagnostic;
class StringRef;
} // end namespace llvm

namespace mlir {
class Location;
class ModuleOp;
class MLIRContext;
class Type;

/// This parses the file specified by the indicated SourceMgr and returns an
/// MLIR module if it was valid.  If not, the error message is emitted through
/// the error handler registered in the context, and a null pointer is returned.
ModuleOp parseSourceFile(const llvm::SourceMgr &sourceMgr,
                         MLIRContext *context);

/// This parses the file specified by the indicated filename and returns an
/// MLIR module if it was valid.  If not, the error message is emitted through
/// the error handler registered in the context, and a null pointer is returned.
ModuleOp parseSourceFile(llvm::StringRef filename, MLIRContext *context);

/// This parses the file specified by the indicated filename using the provided
/// SourceMgr and returns an MLIR module if it was valid.  If not, the error
/// message is emitted through the error handler registered in the context, and
/// a null pointer is returned.
ModuleOp parseSourceFile(llvm::StringRef filename, llvm::SourceMgr &sourceMgr,
                         MLIRContext *context);

/// This parses the module string to a MLIR module if it was valid.  If not, the
/// error message is emitted through the error handler registered in the
/// context, and a null pointer is returned.
ModuleOp parseSourceString(llvm::StringRef moduleStr, MLIRContext *context);

/// This parses a single MLIR type to an MLIR context if it was valid.  If not,
/// an error message is emitted through a new SourceMgrDiagnosticHandler
/// constructed from a new SourceMgr with a single a MemoryBuffer wrapping
/// `typeStr`. If the passed `typeStr` has additional tokens that were not part
/// of the type, an error is emitted.
// TODO(ntv) Improve diagnostic reporting.
Type parseType(llvm::StringRef typeStr, MLIRContext *context);

} // end namespace mlir

#endif // MLIR_PARSER_H
