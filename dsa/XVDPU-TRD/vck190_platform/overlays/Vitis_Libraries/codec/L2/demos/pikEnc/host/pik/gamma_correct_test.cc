// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/gamma_correct.h"

#include <numeric>

#include "gtest/gtest.h"

namespace pik {
namespace {

TEST(GammaCorrectTest, TestLinearToSrgbEdgeCases) {
  EXPECT_EQ(0, LinearToSrgb8Direct(0.0));
  EXPECT_NEAR(0, LinearToSrgb8Direct(1E-6f), 2E-5);
  EXPECT_EQ(0, LinearToSrgb8Direct(-1E-6f));
  EXPECT_EQ(0, LinearToSrgb8Direct(-1E6));
  EXPECT_NEAR(255, LinearToSrgb8Direct(255 - 1E-6f), 1E-5);
  EXPECT_EQ(255, LinearToSrgb8Direct(255 + 1E-6f));
  EXPECT_EQ(255, LinearToSrgb8Direct(1E6));
}

TEST(GammaCorrectTest, TestRoundTrip) {
  double max_err = 0.0;
  for (double linear = 0.0; linear <= 255.0; linear += 1E-4) {
    const double srgb = LinearToSrgb8Direct(linear);
    const double linear2 = Srgb8ToLinearDirect(srgb);
    max_err = std::max(max_err, std::abs(linear - linear2));
  }
  EXPECT_LT(max_err, 2E-13);
}

}  // namespace
}  // namespace pik
