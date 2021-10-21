// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/fields.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>

#include "gtest/gtest.h"

namespace pik {
namespace {

void TestU64Coder(uint64_t value, size_t expected_size) {
  U64Coder coder;

  std::vector<uint8_t> array(1000, 0);
  uint8_t* storage = array.data();

  size_t pos = 0;

  size_t precheck_pos;
  EXPECT_TRUE(coder.CanEncode(value, &precheck_pos));
  EXPECT_EQ(expected_size, precheck_pos);

  EXPECT_TRUE(coder.Write(value, &pos, storage));
  EXPECT_EQ(expected_size, pos);

  BitReader br(storage, array.size());
  uint64_t decoded_value = coder.Read(&br);

  EXPECT_EQ(value, decoded_value);
}

TEST(FieldsTest, U64CoderTest) {
  // Values that should take 2 bits (selector 00): 0
  TestU64Coder(0, 2);

  // Values that should take 6 bits (2 for selector, 4 for value): 1..16
  TestU64Coder(1, 6);
  TestU64Coder(2, 6);
  TestU64Coder(8, 6);
  TestU64Coder(15, 6);
  TestU64Coder(16, 6);

  // Values that should take 10 bits (2 for selector, 8 for value): 17..272
  TestU64Coder(17, 10);
  TestU64Coder(18, 10);
  TestU64Coder(100, 10);
  TestU64Coder(271, 10);
  TestU64Coder(272, 10);

  // Values that should take 15 bits (2 for selector, 12 for value, 1 for varint
  // end): (0)..273..4095
  TestU64Coder(273, 15);
  TestU64Coder(274, 15);
  TestU64Coder(1000, 15);
  TestU64Coder(4094, 15);
  TestU64Coder(4095, 15);

  // Take 24 bits (of which 20 actual value): (0)..4096..1048575
  TestU64Coder(4096, 24);
  TestU64Coder(4097, 24);
  TestU64Coder(10000, 24);
  TestU64Coder(1048574, 24);
  TestU64Coder(1048575, 24);

  // Take 33 bits (of which 28 actual value): (0)..1048576..268435455
  TestU64Coder(1048576, 33);
  TestU64Coder(1048577, 33);
  TestU64Coder(10000000, 33);
  TestU64Coder(268435454, 33);
  TestU64Coder(268435455, 33);

  // Take 42 bits (of which 36 actual value): (0)..268435456..68719476735
  TestU64Coder(268435456ull, 42);
  TestU64Coder(268435457ull, 42);
  TestU64Coder(1000000000ull, 42);
  TestU64Coder(68719476734ull, 42);
  TestU64Coder(68719476735ull, 42);

  // Take 51 bits (of which 44 actual value): (0)..68719476736..17592186044415
  TestU64Coder(68719476736ull, 51);
  TestU64Coder(68719476737ull, 51);
  TestU64Coder(1000000000000ull, 51);
  TestU64Coder(17592186044414ull, 51);
  TestU64Coder(17592186044415ull, 51);

  // Take 60 bits (of which 52 actual value):
  // (0)..17592186044416..4503599627370495
  TestU64Coder(17592186044416ull, 60);
  TestU64Coder(17592186044417ull, 60);
  TestU64Coder(100000000000000ull, 60);
  TestU64Coder(4503599627370494ull, 60);
  TestU64Coder(4503599627370495ull, 60);

  // Take 69 bits (of which 60 actual value):
  // (0)..4503599627370496..1152921504606846975
  TestU64Coder(4503599627370496ull, 69);
  TestU64Coder(4503599627370497ull, 69);
  TestU64Coder(10000000000000000ull, 69);
  TestU64Coder(1152921504606846974ull, 69);
  TestU64Coder(1152921504606846975ull, 69);

  // Take 73 bits (of which 64 actual value):
  // (0)..1152921504606846976..18446744073709551615
  TestU64Coder(1152921504606846976ull, 73);
  TestU64Coder(1152921504606846977ull, 73);
  TestU64Coder(10000000000000000000ull, 73);
  TestU64Coder(18446744073709551614ull, 73);
  TestU64Coder(18446744073709551615ull, 73);
}

}  // namespace
}  // namespace pik
