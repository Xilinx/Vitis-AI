// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/bits.h"
#include "gtest/gtest.h"

namespace pik {
namespace {

TEST(BitsTest, TestPopCount) {
  EXPECT_EQ(0, PopCount(0));
  EXPECT_EQ(1, PopCount(1));
  EXPECT_EQ(1, PopCount(2));
  EXPECT_EQ(2, PopCount(3));
  EXPECT_EQ(1, PopCount(0x80000000u));
  EXPECT_EQ(31, PopCount(0x7FFFFFFF));
  EXPECT_EQ(32, PopCount(0xFFFFFFFF));
}

uint64_t Make64(const uint64_t x) { return x; }

TEST(BitsTest, TestNumZeroBits) {
  // Zero input is well-defined.
  EXPECT_EQ(32, NumZeroBitsAboveMSB(0u));
  EXPECT_EQ(64, NumZeroBitsAboveMSB(Make64(0)));
  EXPECT_EQ(32, NumZeroBitsBelowLSB(0u));
  EXPECT_EQ(64, NumZeroBitsBelowLSB(Make64(0)));

  EXPECT_EQ(31, NumZeroBitsAboveMSB(1u));
  EXPECT_EQ(30, NumZeroBitsAboveMSB(2u));
  EXPECT_EQ(63, NumZeroBitsAboveMSB(Make64(1)));
  EXPECT_EQ(62, NumZeroBitsAboveMSB(Make64(2)));

  EXPECT_EQ(0, NumZeroBitsBelowLSB(1u));
  EXPECT_EQ(0, NumZeroBitsBelowLSB(Make64(1)));
  EXPECT_EQ(1, NumZeroBitsBelowLSB(2u));
  EXPECT_EQ(1, NumZeroBitsBelowLSB(Make64(2)));

  EXPECT_EQ(0, NumZeroBitsAboveMSB(0x80000000U));
  EXPECT_EQ(0, NumZeroBitsAboveMSB(Make64(0x8000000000000000ULL)));
  EXPECT_EQ(31, NumZeroBitsBelowLSB(0x80000000U));
  EXPECT_EQ(63, NumZeroBitsBelowLSB(Make64(0x8000000000000000ULL)));
}

TEST(BitsTest, TestFloorLog2) {
  // for input = [1, 7]
  const int expected[7] = {0, 1, 1, 2, 2, 2, 2};
  for (uint32_t i = 1; i <= 7; ++i) {
    EXPECT_EQ(expected[i - 1], FloorLog2Nonzero(i)) << " " << i;
    EXPECT_EQ(expected[i - 1], FloorLog2Nonzero(Make64(i))) << " " << i;
  }

  EXPECT_EQ(31, FloorLog2Nonzero(0x80000000u));
  EXPECT_EQ(31, FloorLog2Nonzero(0x80000001u));
  EXPECT_EQ(31, FloorLog2Nonzero(0xFFFFFFFFu));

  EXPECT_EQ(31, FloorLog2Nonzero(Make64(0x80000000ull)));
  EXPECT_EQ(31, FloorLog2Nonzero(Make64(0x80000001ull)));
  EXPECT_EQ(31, FloorLog2Nonzero(Make64(0xFFFFFFFFull)));

  EXPECT_EQ(63, FloorLog2Nonzero(Make64(0x8000000000000000ull)));
  EXPECT_EQ(63, FloorLog2Nonzero(Make64(0x8000000000000001ull)));
  EXPECT_EQ(63, FloorLog2Nonzero(Make64(0xFFFFFFFFFFFFFFFFull)));
}

TEST(BitsTest, TestCeilLog2) {
  // for input = [1, 7]
  const int expected[7] = {0, 1, 2, 2, 3, 3, 3};
  for (uint32_t i = 1; i <= 7; ++i) {
    EXPECT_EQ(expected[i - 1], CeilLog2Nonzero(i)) << " " << i;
    EXPECT_EQ(expected[i - 1], CeilLog2Nonzero(Make64(i))) << " " << i;
  }

  EXPECT_EQ(31, CeilLog2Nonzero(0x80000000u));
  EXPECT_EQ(32, CeilLog2Nonzero(0x80000001u));
  EXPECT_EQ(32, CeilLog2Nonzero(0xFFFFFFFFu));

  EXPECT_EQ(31, CeilLog2Nonzero(Make64(0x80000000ull)));
  EXPECT_EQ(32, CeilLog2Nonzero(Make64(0x80000001ull)));
  EXPECT_EQ(32, CeilLog2Nonzero(Make64(0xFFFFFFFFull)));

  EXPECT_EQ(63, CeilLog2Nonzero(Make64(0x8000000000000000ull)));
  EXPECT_EQ(64, CeilLog2Nonzero(Make64(0x8000000000000001ull)));
  EXPECT_EQ(64, CeilLog2Nonzero(Make64(0xFFFFFFFFFFFFFFFFull)));
}

}  // namespace
}  // namespace pik
