// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/byte_order.h"
#include "gtest/gtest.h"

namespace pik {
namespace {

TEST(ByteOrderTest, TestRoundTripBE16) {
  const uint32_t in = 0x1234;
  uint8_t buf[2];
  StoreBE16(in, buf);
  EXPECT_EQ(in, LoadBE16(buf));
  EXPECT_NE(in, LoadLE16(buf));
}

TEST(ByteOrderTest, TestRoundTripLE16) {
  const uint32_t in = 0x1234;
  uint8_t buf[2];
  StoreLE16(in, buf);
  EXPECT_EQ(in, LoadLE16(buf));
  EXPECT_NE(in, LoadBE16(buf));
}

TEST(ByteOrderTest, TestRoundTripBE32) {
  const uint32_t in = 0xFEDCBA98u;
  uint8_t buf[4];
  StoreBE32(in, buf);
  EXPECT_EQ(in, LoadBE32(buf));
  EXPECT_NE(in, LoadLE32(buf));
}

TEST(ByteOrderTest, TestRoundTripLE32) {
  const uint32_t in = 0xFEDCBA98u;
  uint8_t buf[4];
  StoreLE32(in, buf);
  EXPECT_EQ(in, LoadLE32(buf));
  EXPECT_NE(in, LoadBE32(buf));
}

}  // namespace
}  // namespace pik
