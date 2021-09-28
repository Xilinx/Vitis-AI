// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// Facade for calling all enabled and supported SimdTest instantiations.

#include "pik/simd/simd.h"
#include <stdio.h>

namespace pik {
namespace {

// Specialized in simd_test.cctest. Called via TargetBitfield. Thread-hostile.
struct SimdTest {
  template <class Target>
  void operator()();
};

// Called by simd_test.cctest functions.
void NotifyFailure(const int target, const int line, const char* vec_name,
                   const int lane, const char* expected, const char* actual) {
  fprintf(stderr,
          "target %x, line %d, %s lane %d mismatch: expected '%s', got '%s'.\n",
          target, line, vec_name, lane, expected, actual);
  SIMD_TRAP();
}

int RunTests() {
  TargetBitfield().Foreach(SimdTest());
  printf("Successfully tested instruction sets: 0x%x.\n", SIMD_ENABLE);
  return 0;
}

}  // namespace
}  // namespace pik

// Must include "normally" so the build system understands the dependency.
#include "pik/simd/simd_test.cctest"

#define SIMD_ATTR_IMPL "simd_test.cctest"
#include "foreach_target.h"

int main() {
  setvbuf(stdin, nullptr, _IONBF, 0);
  return pik::RunTests();
}
