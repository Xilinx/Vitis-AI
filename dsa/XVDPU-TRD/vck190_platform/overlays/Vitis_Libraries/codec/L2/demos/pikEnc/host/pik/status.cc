// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/status.h"

#include <stdarg.h>
#include <string>

namespace pik {

bool Abort(const char* f, int l, const char* format, ...) {
  char buf[2000];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  const std::string call_stack;

  fprintf(stderr, "Abort at %s:%d: %s\n%s\n", f, l, buf, call_stack.c_str());
  exit(1);
  return false;
}

void Warning(const char* format, ...) {
  char buf[2000];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);
  fprintf(stderr, "%s\n", buf);
}

}  // namespace pik
