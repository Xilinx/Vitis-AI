// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/deconvolve.h"

#include "gtest/gtest.h"

namespace pik {
namespace {

TEST(DeconvolveTest, Trivial) {
  float filter[1] = {1.0};
  float inverse_filter[1];
  EXPECT_LE(InvertConvolution(filter, 1, inverse_filter, 1), 1e-6);
  EXPECT_NEAR(inverse_filter[0], 1.0, 1e-9);
}

TEST(DeconvolveTest, Trivial3) {
  float filter[3] = {0.0, 1.0, 0.0};
  float inverse_filter[3];
  EXPECT_LE(InvertConvolution(filter, 3, inverse_filter, 3), 1e-6);
}

TEST(DeconvolveTest, Infeasible) {
  float filter[3] = {1.0, 0.0, 1.0};
  float inverse_filter[1];
  EXPECT_GE(InvertConvolution(filter, 3, inverse_filter, 1), 0.9);
}

TEST(DeconvolveTest, Singular) {
  float filter[3] = {1.0, 0.0, -1.0};
  float inverse_filter[3];
  // TODO(robryk): Shouldn't this be worse, compared to others?
  EXPECT_GE(InvertConvolution(filter, 3, inverse_filter, 3), 0.3);
}

TEST(DeconvolveTest, Box) {
  float filter[3] = {1.0, 1.0, 1.0};
  float inverse_filter[5];
  EXPECT_LE(InvertConvolution(filter, 3, inverse_filter, 5), 0.26);
}

TEST(DeconvolveTest, Smoke) {
  float filter[3] = {0.5, 1.0, 0.5};
  float inverse_filter[31];
  EXPECT_LE(InvertConvolution(filter, 3, inverse_filter, 3), 0.21);
  EXPECT_LE(InvertConvolution(filter, 3, inverse_filter, 5), 0.15);
  EXPECT_LE(InvertConvolution(filter, 3, inverse_filter, 7), 0.12);
  EXPECT_LE(InvertConvolution(filter, 3, inverse_filter, 31), 0.05);
}

TEST(DeconvolveTest, Asymmetric) {
  float filter[3] = {1.0, 0.0, 0.0};
  float inverse_filter[3];
  EXPECT_LE(InvertConvolution(filter, 3, inverse_filter, 3), 1e-3);
}

}  // namespace
}  // namespace pik
