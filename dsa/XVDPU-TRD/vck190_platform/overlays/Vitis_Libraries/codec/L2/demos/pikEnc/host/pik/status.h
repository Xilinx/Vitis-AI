// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_STATUS_H_
#define PIK_STATUS_H_

// Error handling: Status return type + helper macros.

#include <cstdio>
#include <cstdlib>

#include "pik/compiler_specific.h"

namespace pik {

#ifndef PIK_ENABLE_ASSERT
#define PIK_ENABLE_ASSERT 1
#endif

// Exits the program after printing file/line plus a formatted string.
PIK_FORMAT(3, 4) bool Abort(const char* f, int l, const char* format, ...);

// Emits a warning to standard error. Will be replaced with proper error
// reporting in the future.
PIK_FORMAT(1, 2) void Warning(const char* format, ...);

#define PIK_ABORT(...) Abort(__FILE__, __LINE__, __VA_ARGS__)

// Does not guarantee running the code, use only for debug mode checks.
#if PIK_ENABLE_ASSERT || defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)
#define PIK_ASSERT(condition)                           \
  while (!(condition)) {                                \
    Abort(__FILE__, __LINE__, "Assert %s", #condition); \
  }
#else
#define PIK_ASSERT(condition)
#endif

// Always runs the condition, so can be used for non-debug calls.
#define PIK_CHECK(condition)                           \
  while (!(condition)) {                               \
    Abort(__FILE__, __LINE__, "Check %s", #condition); \
  }

// Always runs the condition, so can be used for non-debug calls.
#define PIK_RETURN_IF_ERROR(condition) \
  while (!(condition)) return false

// Annotation for the location where an error condition is first noticed.
// Error codes are too unspecific to pinpoint the exact location, so we
// add a build flag that crashes and dumps stack at the actual error source.
#ifdef PIK_CRASH_ON_ERROR
#define PIK_NOTIFY_ERROR(message_string) \
  (void)Abort(__FILE__, __LINE__, message_string)
#define PIK_FAILURE(...) Abort(__FILE__, __LINE__, __VA_ARGS__)
#else
#define PIK_NOTIFY_ERROR(message_string)
#define PIK_FAILURE(...) false
#endif

// Drop-in replacement for bool that raises compiler warnings if not used
// after being returned from a function. Example:
// Status LoadFile(...) { return true; } is more compact than
// bool PIK_MUST_USE_RESULT LoadFile(...) { return true; }
class PIK_MUST_USE_RESULT Status {
 public:
  Status(bool ok) : ok_(ok) {}

  operator bool() const { return ok_; }

 private:
  bool ok_;
};

}  // namespace pik

#endif  // PIK_STATUS_H_
