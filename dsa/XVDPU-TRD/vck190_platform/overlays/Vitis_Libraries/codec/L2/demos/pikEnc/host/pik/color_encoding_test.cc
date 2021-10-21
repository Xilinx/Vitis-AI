// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "gtest/gtest.h"

#include "pik/color_encoding.h"

namespace pik {
namespace {

TEST(ColorEncodingTest, RoundTripWhitePoint) {
  for (WhitePoint wp : Values(WhitePoint::kUnknown)) {
    CIExy xy;
    EXPECT_TRUE(WhitePointToCIExy(wp, &xy));
    EXPECT_EQ(wp, WhitePointFromCIExy(xy));
  }
}

TEST(ColorEncodingTest, RoundTripPrimaries) {
  for (Primaries pr : Values(Primaries::kUnknown)) {
    PrimariesCIExy xy;
    EXPECT_TRUE(PrimariesToCIExy(pr, &xy));
    EXPECT_EQ(pr, PrimariesFromCIExy(xy));
  }
}

TEST(ColorEncodingTest, RoundTripTransferFunction) {
  const auto unknown = TransferFunction::kUnknown;
  for (TransferFunction tf : Values(unknown)) {
    const double gamma = GammaFromTransferFunction(tf);
    EXPECT_EQ(tf, TransferFunctionFromGamma(gamma));

    const std::string& str = StringFromGamma(gamma);
    double gamma2 = GammaFromString(str);
    EXPECT_EQ(gamma, gamma2);
  }

  // Invalid gamma
  EXPECT_EQ(unknown, TransferFunctionFromGamma(0.0));
  EXPECT_EQ(unknown, TransferFunctionFromGamma(1.0001));
  EXPECT_EQ(unknown, TransferFunctionFromGamma(-1.0));

  // Unusual but valid gamma
  EXPECT_EQ(0.1234, GammaFromString(StringFromGamma(0.1234)));
}

}  // namespace
}  // namespace pik
