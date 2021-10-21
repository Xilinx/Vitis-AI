// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/epf.h"

#include "gtest/gtest.h"
#include "pik/simd/targets.h"

namespace pik {
namespace {

TEST(EdgePreservingFilterTest, TestInternal) {
  // This test requires implementation details of the cc file, which we prefer
  // not to expose in the header due to target-specific namespace.
  TargetBitfield().Foreach(EdgePreservingFilterTest());
}

TEST(EdgePreservingFilterTest, TestWeight) {
  TargetBitfield targets;
  const int sigma = 32 << EdgePreservingFilter::kSigmaShift;
  for (int sad = 0; sad < 100; ++sad) {
    const float weight =
        Dispatch(targets.Best(), EdgePreservingFilterTest(), sigma, sad);
    printf("%d,%.3f\n", sad, weight);
  }
}

}  // namespace
}  // namespace pik
