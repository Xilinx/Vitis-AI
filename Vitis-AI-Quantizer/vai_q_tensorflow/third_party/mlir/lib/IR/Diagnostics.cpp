//===- Diagnostics.cpp - MLIR Diagnostics ---------------------------------===//
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

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// DiagnosticArgument
//===----------------------------------------------------------------------===//

// Construct from an Attribute.
DiagnosticArgument::DiagnosticArgument(Attribute attr)
    : kind(DiagnosticArgumentKind::Attribute),
      opaqueVal(reinterpret_cast<intptr_t>(attr.getAsOpaquePointer())) {}

// Construct from a Type.
DiagnosticArgument::DiagnosticArgument(Type val)
    : kind(DiagnosticArgumentKind::Type),
      opaqueVal(reinterpret_cast<intptr_t>(val.getAsOpaquePointer())) {}

/// Returns this argument as an Attribute.
Attribute DiagnosticArgument::getAsAttribute() const {
  assert(getKind() == DiagnosticArgumentKind::Attribute);
  return Attribute::getFromOpaquePointer(
      reinterpret_cast<const void *>(opaqueVal));
}

/// Returns this argument as a Type.
Type DiagnosticArgument::getAsType() const {
  assert(getKind() == DiagnosticArgumentKind::Type);
  return Type::getFromOpaquePointer(reinterpret_cast<const void *>(opaqueVal));
}

/// Outputs this argument to a stream.
void DiagnosticArgument::print(raw_ostream &os) const {
  switch (kind) {
  case DiagnosticArgumentKind::Attribute:
    os << getAsAttribute();
    break;
  case DiagnosticArgumentKind::Double:
    os << getAsDouble();
    break;
  case DiagnosticArgumentKind::Integer:
    os << getAsInteger();
    break;
  case DiagnosticArgumentKind::Operation:
    os << getAsOperation();
    break;
  case DiagnosticArgumentKind::String:
    os << getAsString();
    break;
  case DiagnosticArgumentKind::Type:
    os << '\'' << getAsType() << '\'';
    break;
  case DiagnosticArgumentKind::Unsigned:
    os << getAsUnsigned();
    break;
  }
}

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

/// Convert a Twine to a StringRef. Memory used for generating the StringRef is
/// stored in 'strings'.
static StringRef twineToStrRef(const Twine &val,
                               std::vector<std::unique_ptr<char[]>> &strings) {
  // Allocate memory to hold this string.
  llvm::SmallString<64> data;
  auto strRef = val.toStringRef(data);
  strings.push_back(std::unique_ptr<char[]>(new char[strRef.size()]));
  memcpy(&strings.back()[0], strRef.data(), strRef.size());

  // Return a reference to the new string.
  return StringRef(&strings.back()[0], strRef.size());
}

/// Stream in a Twine argument.
Diagnostic &Diagnostic::operator<<(char val) { return *this << Twine(val); }
Diagnostic &Diagnostic::operator<<(const Twine &val) {
  arguments.push_back(DiagnosticArgument(twineToStrRef(val, strings)));
  return *this;
}
Diagnostic &Diagnostic::operator<<(Twine &&val) {
  arguments.push_back(DiagnosticArgument(twineToStrRef(val, strings)));
  return *this;
}

/// Stream in an Identifier.
Diagnostic &Diagnostic::operator<<(Identifier val) {
  // An identifier is stored in the context, so we don't need to worry about the
  // lifetime of its data.
  arguments.push_back(DiagnosticArgument(val.strref()));
  return *this;
}

/// Stream in an OperationName.
Diagnostic &Diagnostic::operator<<(OperationName val) {
  // An OperationName is stored in the context, so we don't need to worry about
  // the lifetime of its data.
  arguments.push_back(DiagnosticArgument(val.getStringRef()));
  return *this;
}

/// Outputs this diagnostic to a stream.
void Diagnostic::print(raw_ostream &os) const {
  for (auto &arg : getArguments())
    arg.print(os);
}

/// Convert the diagnostic to a string.
std::string Diagnostic::str() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

/// Attaches a note to this diagnostic. A new location may be optionally
/// provided, if not, then the location defaults to the one specified for this
/// diagnostic. Notes may not be attached to other notes.
Diagnostic &Diagnostic::attachNote(llvm::Optional<Location> noteLoc) {
  // We don't allow attaching notes to notes.
  assert(severity != DiagnosticSeverity::Note &&
         "cannot attach a note to a note");

  // If a location wasn't provided then reuse our location.
  if (!noteLoc)
    noteLoc = loc;

  /// Append and return a new note.
  notes.push_back(
      std::make_unique<Diagnostic>(*noteLoc, DiagnosticSeverity::Note));
  return *notes.back();
}

/// Allow a diagnostic to be converted to 'failure'.
Diagnostic::operator LogicalResult() const { return failure(); }

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

/// Allow an inflight diagnostic to be converted to 'failure', otherwise
/// 'success' if this is an empty diagnostic.
InFlightDiagnostic::operator LogicalResult() const {
  return failure(isActive());
}

/// Reports the diagnostic to the engine.
void InFlightDiagnostic::report() {
  // If this diagnostic is still inflight and it hasn't been abandoned, then
  // report it.
  if (isInFlight()) {
    owner->emit(std::move(*impl));
    owner = nullptr;
  }
  impl.reset();
}

/// Abandons this diagnostic.
void InFlightDiagnostic::abandon() { owner = nullptr; }

//===----------------------------------------------------------------------===//
// DiagnosticEngineImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct DiagnosticEngineImpl {
  /// Emit a diagnostic using the registered issue handle if present, or with
  /// the default behavior if not.
  void emit(Diagnostic diag);

  /// A mutex to ensure that diagnostics emission is thread-safe.
  llvm::sys::SmartMutex<true> mutex;

  /// This is the handler to use to report diagnostics, or null if not
  /// registered.
  DiagnosticEngine::HandlerTy handler;
};
} // namespace detail
} // namespace mlir

/// Emit a diagnostic using the registered issue handle if present, or with
/// the default behavior if not.
void DiagnosticEngineImpl::emit(Diagnostic diag) {
  llvm::sys::SmartScopedLock<true> lock(mutex);

  // If we had a handler registered, emit the diagnostic using it.
  if (handler)
    return handler(std::move(diag));

  // Otherwise, if this is an error we emit it to stderr.
  if (diag.getSeverity() != DiagnosticSeverity::Error)
    return;

  auto &os = llvm::errs();
  if (!diag.getLocation().isa<UnknownLoc>())
    os << diag.getLocation() << ": ";
  os << "error: ";

  // The default behavior for errors is to emit them to stderr.
  os << diag << '\n';
  os.flush();
}

//===----------------------------------------------------------------------===//
// DiagnosticEngine
//===----------------------------------------------------------------------===//

DiagnosticEngine::DiagnosticEngine() : impl(new DiagnosticEngineImpl()) {}
DiagnosticEngine::~DiagnosticEngine() {}

/// Set the diagnostic handler for this engine.  The handler is passed
/// location information if present (nullptr if not) along with a message and
/// a severity that indicates whether this is an error, warning, etc. Note
/// that this replaces any existing handler.
void DiagnosticEngine::setHandler(const HandlerTy &handler) {
  impl->handler = handler;
}

/// Return the current diagnostic handler, or null if none is present.
auto DiagnosticEngine::getHandler() -> HandlerTy {
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  return impl->handler;
}

/// Emit a diagnostic using the registered issue handler if present, or with
/// the default behavior if not.
void DiagnosticEngine::emit(Diagnostic diag) {
  assert(diag.getSeverity() != DiagnosticSeverity::Note &&
         "notes should not be emitted directly");
  impl->emit(std::move(diag));
}

/// Helper function used to emit a diagnostic with an optionally empty twine
/// message. If the message is empty, then it is not inserted into the
/// diagnostic.
static InFlightDiagnostic emitDiag(Location location,
                                   DiagnosticSeverity severity,
                                   const llvm::Twine &message) {
  auto &diagEngine = location->getContext()->getDiagEngine();
  auto diag = diagEngine.emit(location, severity);
  if (!message.isTriviallyEmpty())
    diag << message;
  return diag;
}

/// Emit an error message using this location.
InFlightDiagnostic mlir::emitError(Location loc) { return emitError(loc, {}); }
InFlightDiagnostic mlir::emitError(Location loc, const Twine &message) {
  return emitDiag(loc, DiagnosticSeverity::Error, message);
}

/// Emit a warning message using this location.
InFlightDiagnostic mlir::emitWarning(Location loc) {
  return emitWarning(loc, {});
}
InFlightDiagnostic mlir::emitWarning(Location loc, const Twine &message) {
  return emitDiag(loc, DiagnosticSeverity::Warning, message);
}

/// Emit a remark message using this location.
InFlightDiagnostic mlir::emitRemark(Location loc) {
  return emitRemark(loc, {});
}
InFlightDiagnostic mlir::emitRemark(Location loc, const Twine &message) {
  return emitDiag(loc, DiagnosticSeverity::Remark, message);
}

//===----------------------------------------------------------------------===//
// ScopedDiagnosticHandler
//===----------------------------------------------------------------------===//

ScopedDiagnosticHandler::ScopedDiagnosticHandler(MLIRContext *ctx)
    : existingHandler(ctx->getDiagEngine().getHandler()), ctx(ctx) {}
ScopedDiagnosticHandler::ScopedDiagnosticHandler(
    MLIRContext *ctx, const DiagnosticEngine::HandlerTy &handler)
    : ScopedDiagnosticHandler(ctx) {
  ctx->getDiagEngine().setHandler(handler);
}
ScopedDiagnosticHandler::~ScopedDiagnosticHandler() {
  ctx->getDiagEngine().setHandler(existingHandler);
}

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticHandler
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {
struct SourceMgrDiagnosticHandlerImpl {
  /// Get a memory buffer for the given file, or nullptr if one is not found.
  const llvm::MemoryBuffer *getBufferForFile(llvm::SourceMgr &mgr,
                                             StringRef filename) {
    // Check for an existing mapping to the buffer id for this file.
    auto bufferIt = filenameToBuf.find(filename);
    if (bufferIt != filenameToBuf.end())
      return bufferIt->second;

    // Look for a buffer in the manager that has this filename.
    for (unsigned i = 1, e = mgr.getNumBuffers() + 1; i != e; ++i) {
      auto *buf = mgr.getMemoryBuffer(i);
      if (buf->getBufferIdentifier() == filename)
        return filenameToBuf[filename] = buf;
    }

    // Otherwise, try to load the source file.
    const llvm::MemoryBuffer *newBuf = nullptr;
    std::string ignored;
    if (auto newBufID = mgr.AddIncludeFile(filename, llvm::SMLoc(), ignored))
      newBuf = mgr.getMemoryBuffer(newBufID);
    return filenameToBuf[filename] = newBuf;
  }

  /// Mapping between file name and buffer pointer.
  llvm::StringMap<const llvm::MemoryBuffer *> filenameToBuf;
};
} // end namespace detail
} // end namespace mlir

/// Return a processable FileLineColLoc from the given location.
static llvm::Optional<FileLineColLoc> getFileLineColLoc(Location loc) {
  switch (loc->getKind()) {
  case StandardAttributes::NameLocation:
    return getFileLineColLoc(loc.cast<NameLoc>().getChildLoc());
  case StandardAttributes::FileLineColLocation:
    return loc.cast<FileLineColLoc>();
  case StandardAttributes::CallSiteLocation:
    // Process the callee of a callsite location.
    return getFileLineColLoc(loc.cast<CallSiteLoc>().getCallee());
  default:
    return llvm::None;
  }
}

/// Given a diagnostic kind, returns the LLVM DiagKind.
static llvm::SourceMgr::DiagKind getDiagKind(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return llvm::SourceMgr::DK_Note;
  case DiagnosticSeverity::Warning:
    return llvm::SourceMgr::DK_Warning;
  case DiagnosticSeverity::Error:
    return llvm::SourceMgr::DK_Error;
  case DiagnosticSeverity::Remark:
    return llvm::SourceMgr::DK_Remark;
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

SourceMgrDiagnosticHandler::SourceMgrDiagnosticHandler(llvm::SourceMgr &mgr,
                                                       MLIRContext *ctx,
                                                       llvm::raw_ostream &os)
    : ScopedDiagnosticHandler(ctx), mgr(mgr), os(os),
      impl(new SourceMgrDiagnosticHandlerImpl()) {
  // Register a simple diagnostic handler.
  ctx->getDiagEngine().setHandler(
      [this](Diagnostic diag) { emitDiagnostic(diag); });
}

SourceMgrDiagnosticHandler::SourceMgrDiagnosticHandler(llvm::SourceMgr &mgr,
                                                       MLIRContext *ctx)
    : SourceMgrDiagnosticHandler(mgr, ctx, llvm::errs()) {}

SourceMgrDiagnosticHandler::~SourceMgrDiagnosticHandler() {}

void SourceMgrDiagnosticHandler::emitDiagnostic(Location loc, Twine message,
                                                DiagnosticSeverity kind) {
  // Extract a file location from this loc.
  auto fileLoc = getFileLineColLoc(loc);

  // If one doesn't exist, then print the raw message without a source location.
  if (!fileLoc) {
    std::string str;
    llvm::raw_string_ostream strOS(str);
    if (!loc.isa<UnknownLoc>())
      strOS << loc << ": ";
    strOS << message;
    return mgr.PrintMessage(os, llvm::SMLoc(), getDiagKind(kind), strOS.str());
  }

  // Otherwise, try to convert the file location to an SMLoc.
  auto smloc = convertLocToSMLoc(*fileLoc);
  if (smloc.isValid())
    return mgr.PrintMessage(os, smloc, getDiagKind(kind), message);

  // If the conversion was unsuccessful, create a diagnostic with the file
  // information.
  llvm::SMDiagnostic diag(fileLoc->getFilename(), getDiagKind(kind),
                          message.str());
  diag.print(nullptr, os);
}

/// Emit the given diagnostic with the held source manager.
void SourceMgrDiagnosticHandler::emitDiagnostic(Diagnostic &diag) {
  // Emit the diagnostic.
  auto loc = diag.getLocation();
  emitDiagnostic(loc, diag.str(), diag.getSeverity());

  // If the diagnostic location was a call site location, then print the call
  // stack as well.
  if (auto callLoc = loc.dyn_cast<CallSiteLoc>()) {
    // Print the call stack while valid, or until the limit is reached.
    Location callerLoc = callLoc.getCaller();
    for (unsigned curDepth = 0; curDepth < callStackLimit; ++curDepth) {
      emitDiagnostic(callerLoc, "called from", DiagnosticSeverity::Note);
      if ((callLoc = callerLoc.dyn_cast<CallSiteLoc>()))
        callerLoc = callLoc.getCaller();
      else
        break;
    }
  }

  // Emit each of the notes.
  for (auto &note : diag.getNotes())
    emitDiagnostic(note.getLocation(), note.str(), note.getSeverity());
}

/// Get a memory buffer for the given file, or nullptr if one is not found.
const llvm::MemoryBuffer *
SourceMgrDiagnosticHandler::getBufferForFile(StringRef filename) {
  return impl->getBufferForFile(mgr, filename);
}

/// Get a memory buffer for the given file, or the main file of the source
/// manager if one doesn't exist. This always returns non-null.
llvm::SMLoc SourceMgrDiagnosticHandler::convertLocToSMLoc(FileLineColLoc loc) {
  // Get the buffer for this filename.
  auto *membuf = getBufferForFile(loc.getFilename());
  if (!membuf)
    return llvm::SMLoc();

  // TODO: This should really be upstreamed to be a method on llvm::SourceMgr.
  // Doing so would allow it to use the offset cache that is already maintained
  // by SrcBuffer, making this more efficient.
  unsigned lineNo = loc.getLine();
  unsigned columnNo = loc.getColumn();

  // Scan for the correct line number.
  const char *position = membuf->getBufferStart();
  const char *end = membuf->getBufferEnd();

  // We start counting line and column numbers from 1.
  if (lineNo != 0)
    --lineNo;
  if (columnNo != 0)
    --columnNo;

  while (position < end && lineNo) {
    auto curChar = *position++;

    // Scan for newlines.  If this isn't one, ignore it.
    if (curChar != '\r' && curChar != '\n')
      continue;

    // We saw a line break, decrement our counter.
    --lineNo;

    // Check for \r\n and \n\r and treat it as a single escape.  We know that
    // looking past one character is safe because MemoryBuffer's are always nul
    // terminated.
    if (*position != curChar && (*position == '\r' || *position == '\n'))
      ++position;
  }

  // If the line/column counter was invalid, return a pointer to the start of
  // the buffer.
  if (lineNo || position + columnNo > end)
    return llvm::SMLoc::getFromPointer(membuf->getBufferStart());

  // If the column is zero, try to skip to the first non-whitespace character.
  if (columnNo == 0) {
    auto isNewline = [](char c) { return c == '\n' || c == '\r'; };
    auto isWhitespace = [](char c) { return c == ' ' || c == '\t'; };

    // Look for a valid non-whitespace character before the next line.
    for (auto *newPos = position; newPos < end && !isNewline(*newPos); ++newPos)
      if (!isWhitespace(*newPos))
        return llvm::SMLoc::getFromPointer(newPos);
  }

  // Otherwise return the right pointer.
  return llvm::SMLoc::getFromPointer(position + columnNo);
}

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticVerifierHandler
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
// Record the expected diagnostic's position, substring and whether it was
// seen.
struct ExpectedDiag {
  DiagnosticSeverity kind;
  unsigned lineNo;
  StringRef substring;
  llvm::SMLoc fileLoc;
  bool matched;
};

struct SourceMgrDiagnosticVerifierHandlerImpl {
  SourceMgrDiagnosticVerifierHandlerImpl() : status(success()) {}

  /// Returns the expected diagnostics for the given source file.
  llvm::Optional<MutableArrayRef<ExpectedDiag>>
  getExpectedDiags(StringRef bufName);

  /// Computes the expected diagnostics for the given source buffer.
  MutableArrayRef<ExpectedDiag>
  computeExpectedDiags(const llvm::MemoryBuffer *buf);

  /// The current status of the verifier.
  LogicalResult status;

  /// A list of expected diagnostics for each buffer of the source manager.
  llvm::StringMap<SmallVector<ExpectedDiag, 2>> expectedDiagsPerFile;

  /// Regex to match the expected diagnostics format.
  llvm::Regex expected = llvm::Regex(
      "expected-(error|note|remark|warning) *(@[+-][0-9]+)? *{{(.*)}}");
};
} // end namespace detail
} // end namespace mlir

/// Given a diagnostic kind, return a human readable string for it.
static StringRef getDiagKindStr(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return "note";
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  case DiagnosticSeverity::Remark:
    return "remark";
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

/// Returns the expected diagnostics for the given source file.
llvm::Optional<MutableArrayRef<ExpectedDiag>>
SourceMgrDiagnosticVerifierHandlerImpl::getExpectedDiags(StringRef bufName) {
  auto expectedDiags = expectedDiagsPerFile.find(bufName);
  if (expectedDiags != expectedDiagsPerFile.end())
    return MutableArrayRef<ExpectedDiag>(expectedDiags->second);
  return llvm::None;
}

/// Computes the expected diagnostics for the given source buffer.
MutableArrayRef<ExpectedDiag>
SourceMgrDiagnosticVerifierHandlerImpl::computeExpectedDiags(
    const llvm::MemoryBuffer *buf) {
  // If the buffer is invalid, return an empty list.
  if (!buf)
    return llvm::None;
  auto &expectedDiags = expectedDiagsPerFile[buf->getBufferIdentifier()];

  // Scan the file for expected-* designators.
  SmallVector<StringRef, 100> lines;
  buf->getBuffer().split(lines, '\n');
  for (unsigned lineNo = 0, e = lines.size(); lineNo < e; ++lineNo) {
    SmallVector<StringRef, 3> matches;
    if (!expected.match(lines[lineNo], &matches))
      continue;
    // Point to the start of expected-*.
    auto expectedStart = llvm::SMLoc::getFromPointer(matches[0].data());

    DiagnosticSeverity kind;
    if (matches[1] == "error")
      kind = DiagnosticSeverity::Error;
    else if (matches[1] == "warning")
      kind = DiagnosticSeverity::Warning;
    else if (matches[1] == "remark")
      kind = DiagnosticSeverity::Remark;
    else {
      assert(matches[1] == "note");
      kind = DiagnosticSeverity::Note;
    }

    ExpectedDiag record{kind, lineNo + 1, matches[3], expectedStart, false};
    auto offsetMatch = matches[2];
    if (!offsetMatch.empty()) {
      int offset;
      // Get the integer value without the @ and +/- prefix.
      if (!offsetMatch.drop_front(2).getAsInteger(0, offset)) {
        if (offsetMatch[1] == '+')
          record.lineNo += offset;
        else
          record.lineNo -= offset;
      }
    }
    expectedDiags.push_back(record);
  }
  return expectedDiags;
}

SourceMgrDiagnosticVerifierHandler::SourceMgrDiagnosticVerifierHandler(
    llvm::SourceMgr &srcMgr, MLIRContext *ctx, llvm::raw_ostream &out)
    : SourceMgrDiagnosticHandler(srcMgr, ctx, out),
      impl(new SourceMgrDiagnosticVerifierHandlerImpl()) {
  // Compute the expected diagnostics for each of the current files in the
  // source manager.
  for (unsigned i = 0, e = mgr.getNumBuffers(); i != e; ++i)
    (void)impl->computeExpectedDiags(mgr.getMemoryBuffer(i + 1));

  // Register a handler to verfy the diagnostics.
  ctx->getDiagEngine().setHandler([&](Diagnostic diag) {
    // Process the main diagnostics.
    process(diag);

    // Process each of the notes.
    for (auto &note : diag.getNotes())
      process(note);
  });
}

SourceMgrDiagnosticVerifierHandler::SourceMgrDiagnosticVerifierHandler(
    llvm::SourceMgr &srcMgr, MLIRContext *ctx)
    : SourceMgrDiagnosticVerifierHandler(srcMgr, ctx, llvm::errs()) {}

SourceMgrDiagnosticVerifierHandler::~SourceMgrDiagnosticVerifierHandler() {
  // Ensure that all expected diagnosics were handled.
  (void)verify();
}

/// Returns the status of the verifier and verifies that all expected
/// diagnostics were emitted. This return success if all diagnostics were
/// verified correctly, failure otherwise.
LogicalResult SourceMgrDiagnosticVerifierHandler::verify() {
  // Verify that all expected errors were seen.
  for (auto &expectedDiagsPair : impl->expectedDiagsPerFile) {
    for (auto &err : expectedDiagsPair.second) {
      if (err.matched)
        continue;
      llvm::SMRange range(err.fileLoc,
                          llvm::SMLoc::getFromPointer(err.fileLoc.getPointer() +
                                                      err.substring.size()));
      mgr.PrintMessage(os, err.fileLoc, llvm::SourceMgr::DK_Error,
                       "expected " + getDiagKindStr(err.kind) + " \"" +
                           err.substring + "\" was not produced",
                       range);
      impl->status = failure();
    }
  }
  impl->expectedDiagsPerFile.clear();
  return impl->status;
}

/// Process a single diagnostic.
void SourceMgrDiagnosticVerifierHandler::process(Diagnostic &diag) {
  auto kind = diag.getSeverity();

  // Process a FileLineColLoc.
  if (auto fileLoc = getFileLineColLoc(diag.getLocation()))
    return process(*fileLoc, diag.str(), kind);

  emitDiagnostic(diag.getLocation(),
                 "unexpected " + getDiagKindStr(kind) + ": " + diag.str(),
                 DiagnosticSeverity::Error);
  impl->status = failure();
}

/// Process a FileLineColLoc diagnostic.
void SourceMgrDiagnosticVerifierHandler::process(FileLineColLoc loc,
                                                 StringRef msg,
                                                 DiagnosticSeverity kind) {
  // Get the expected diagnostics for this file.
  auto diags = impl->getExpectedDiags(loc.getFilename());
  if (!diags)
    diags = impl->computeExpectedDiags(getBufferForFile(loc.getFilename()));

  // Search for a matching expected diagnostic.
  // If we find something that is close then emit a more specific error.
  ExpectedDiag *nearMiss = nullptr;

  // If this was an expected error, remember that we saw it and return.
  unsigned line = loc.getLine();
  for (auto &e : *diags) {
    if (line == e.lineNo && msg.contains(e.substring)) {
      if (e.kind == kind) {
        e.matched = true;
        return;
      }

      // If this only differs based on the diagnostic kind, then consider it
      // to be a near miss.
      nearMiss = &e;
    }
  }

  // Otherwise, emit an error for the near miss.
  if (nearMiss)
    mgr.PrintMessage(os, nearMiss->fileLoc, llvm::SourceMgr::DK_Error,
                     "'" + getDiagKindStr(kind) +
                         "' diagnostic emitted when expecting a '" +
                         getDiagKindStr(nearMiss->kind) + "'");
  else
    emitDiagnostic(loc, "unexpected " + getDiagKindStr(kind) + ": " + msg,
                   DiagnosticSeverity::Error);
  impl->status = failure();
}

//===----------------------------------------------------------------------===//
// ParallelDiagnosticHandler
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct ParallelDiagnosticHandlerImpl : public llvm::PrettyStackTraceEntry {
  struct ThreadDiagnostic {
    ThreadDiagnostic(size_t id, Diagnostic diag)
        : id(id), diag(std::move(diag)) {}
    bool operator<(const ThreadDiagnostic &rhs) const { return id < rhs.id; }

    /// The id for this diagnostic, this is used for ordering.
    /// Note: This id corresponds to the ordered position of the current element
    ///       being processed by a given thread.
    size_t id;

    /// The diagnostic.
    Diagnostic diag;
  };

  ParallelDiagnosticHandlerImpl(MLIRContext *ctx)
      : prevHandler(ctx->getDiagEngine().getHandler()), context(ctx) {
    ctx->getDiagEngine().setHandler([this](Diagnostic diag) {
      uint64_t tid = llvm::get_threadid();
      llvm::sys::SmartScopedLock<true> lock(mutex);
      assert(threadToOrderID.count(tid) &&
             "current thread does not have a valid orderID");

      // Append a new diagnostic.
      diagnostics.emplace_back(threadToOrderID[tid], std::move(diag));
    });
  }

  ~ParallelDiagnosticHandlerImpl() {
    // Restore the previous diagnostic handler.
    context->getDiagEngine().setHandler(prevHandler);

    // Early exit if there are no diagnostics, this is the common case.
    if (diagnostics.empty())
      return;

    // Emit the diagnostics back to the context.
    emitDiagnostics([&](Diagnostic diag) {
      return context->getDiagEngine().emit(std::move(diag));
    });
  }

  /// Utility method to emit any held diagnostics.
  void emitDiagnostics(std::function<void(Diagnostic)> emitFn) {
    // Stable sort all of the diagnostics that were emitted. This creates a
    // deterministic ordering for the diagnostics based upon which order id they
    // were emitted for.
    std::stable_sort(diagnostics.begin(), diagnostics.end());

    // Emit each diagnostic to the context again.
    for (ThreadDiagnostic &diag : diagnostics)
      emitFn(std::move(diag.diag));
  }

  /// Set the order id for the current thread.
  void setOrderIDForThread(size_t orderID) {
    uint64_t tid = llvm::get_threadid();
    llvm::sys::SmartScopedLock<true> lock(mutex);
    threadToOrderID[tid] = orderID;
  }

  /// Dump the current diagnostics that were inflight.
  void print(raw_ostream &os) const override {
    // Early exit if there are no diagnostics, this is the common case.
    if (diagnostics.empty())
      return;

    os << "In-Flight Diagnostics:\n";
    const_cast<ParallelDiagnosticHandlerImpl *>(this)->emitDiagnostics(
        [&](Diagnostic diag) {
          os.indent(4);

          // Print each diagnostic with the format:
          //   "<location>: <kind>: <msg>"
          if (!diag.getLocation().isa<UnknownLoc>())
            os << diag.getLocation() << ": ";
          switch (diag.getSeverity()) {
          case DiagnosticSeverity::Error:
            os << "error: ";
            break;
          case DiagnosticSeverity::Warning:
            os << "warning: ";
            break;
          case DiagnosticSeverity::Note:
            os << "note: ";
            break;
          case DiagnosticSeverity::Remark:
            os << "remark: ";
            break;
          }
          os << diag << '\n';
        });
  }

  /// The previous context diagnostic handler.
  DiagnosticEngine::HandlerTy prevHandler;

  /// A smart mutex to lock access to the internal state.
  llvm::sys::SmartMutex<true> mutex;

  /// A mapping between the thread id and the current order id.
  DenseMap<uint64_t, size_t> threadToOrderID;

  /// An unordered list of diagnostics that were emitted.
  std::vector<ThreadDiagnostic> diagnostics;

  /// The context to emit the diagnostics to.
  MLIRContext *context;
};
} // end namespace detail
} // end namespace mlir

ParallelDiagnosticHandler::ParallelDiagnosticHandler(MLIRContext *ctx)
    : impl(new ParallelDiagnosticHandlerImpl(ctx)) {}
ParallelDiagnosticHandler::~ParallelDiagnosticHandler() {}

/// Set the order id for the current thread.
void ParallelDiagnosticHandler::setOrderIDForThread(size_t orderID) {
  impl->setOrderIDForThread(orderID);
}
