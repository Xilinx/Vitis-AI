// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/padded_bytes.h"
#include "gtest/gtest.h"

namespace pik {
namespace {

TEST(PaddedBytesTest, TestNonEmptyFirstByteZero) {
  PaddedBytes pb(1);
  EXPECT_EQ(0, pb[0]);
  // Even after resizing..
  pb.resize(20);
  EXPECT_EQ(0, pb[0]);
  // And reserving.
  pb.reserve(200);
  EXPECT_EQ(0, pb[0]);
}

TEST(PaddedBytesTest, TestEmptyFirstByteZero) {
  PaddedBytes pb(0);
  // After resizing - new zero is written despite there being nothing to copy.
  pb.resize(20);
  EXPECT_EQ(0, pb[0]);
}

TEST(PaddedBytesTest, TestFillWithoutReserve) {
  PaddedBytes pb;
  for (size_t i = 0; i < 170; ++i) {
    pb.push_back(i);
  }
  EXPECT_EQ(170, pb.size());
  EXPECT_GE(pb.capacity(), 170);
}

TEST(PaddedBytesTest, TestFillWithExactReserve) {
  PaddedBytes pb;
  pb.reserve(170);
  for (size_t i = 0; i < 170; ++i) {
    pb.push_back(i);
  }
  EXPECT_EQ(170, pb.size());
  EXPECT_EQ(pb.capacity(), 170);
}

TEST(PaddedBytesTest, TestFillWithMoreReserve) {
  PaddedBytes pb;
  pb.reserve(171);
  for (size_t i = 0; i < 170; ++i) {
    pb.push_back(i);
  }
  EXPECT_EQ(170, pb.size());
  EXPECT_GT(pb.capacity(), 170);
}

}  // namespace
}  // namespace pik
