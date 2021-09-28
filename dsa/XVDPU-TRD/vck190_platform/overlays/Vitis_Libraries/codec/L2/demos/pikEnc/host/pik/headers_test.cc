// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/headers.h"

#include "gtest/gtest.h"
#include "pik/common.h"
#include "pik/field_encodings.h"
#include "pik/fields.h"

namespace pik {
namespace {

// Ensures Read(Write()) returns the same fields.
TEST(HeadersTest, TestContainer) {
  for (int i = 0; i < 8; i++) {
    FileHeader c;
    c.xsize_minus_1 = 123 + 77 * i;
    c.orientation = static_cast<Orientation>(1 + i);
    c.metadata.transcoded.original_bit_depth = 7 + i;

    size_t extension_bits, total_bits;
    ASSERT_TRUE(CanEncode(c, &extension_bits, &total_bits));
    EXPECT_EQ(0, extension_bits);
    PaddedBytes storage(DivCeil(total_bits, kBitsPerByte));
    size_t pos = 0;

    ASSERT_TRUE(WriteFileHeader(c, extension_bits, &pos, storage.data()));
    EXPECT_EQ(total_bits, pos);

    FileHeader c2;
    BitReader reader(storage.data(), storage.size());
    ASSERT_TRUE(ReadFileHeader(&reader, &c2));
    EXPECT_EQ(total_bits, reader.BitsRead());

    EXPECT_EQ(c.xsize_minus_1, c2.xsize_minus_1);
    EXPECT_EQ(c.metadata.transcoded.original_bit_depth,
              c2.metadata.transcoded.original_bit_depth);
    EXPECT_EQ(c.orientation, c2.orientation);
    // Also equal if default-initialized.
    EXPECT_EQ(c.ysize_minus_1, c2.ysize_minus_1);
    EXPECT_EQ(c.extensions, c2.extensions);
  }
}

// Changing serialized signature causes ReadFileHeader to fail.
#ifndef PIK_CRASH_ON_ERROR
TEST(HeadersTest, TestSignature) {
  FileHeader c;
  size_t extension_bits, total_bits;
  ASSERT_TRUE(CanEncode(c, &extension_bits, &total_bits));
  EXPECT_EQ(0, extension_bits);

  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    uint8_t storage[99];
    ASSERT_LE(total_bits, sizeof(storage) * kBitsPerByte);
    size_t pos = 0;
    ASSERT_TRUE(WriteFileHeader(c, extension_bits, &pos, storage));
    ASSERT_EQ(pos, total_bits);

    storage[i] = 0;
    FileHeader c2;
    BitReader reader(storage, sizeof(storage));
    ASSERT_FALSE(ReadFileHeader(&reader, &c2));
  }
}
#endif

// Ensure maximal values can be stored.
TEST(HeadersTest, TestMaxValue) {
  FrameHeader h;
  h.size = ~0ull;
  h.flags = ~0u;
  size_t extension_bits, total_bits;
  ASSERT_TRUE(CanEncode(h, &extension_bits, &total_bits));
  EXPECT_EQ(0, extension_bits);
  PaddedBytes storage(DivCeil(total_bits, kBitsPerByte));
  size_t pos = 0;
  ASSERT_TRUE(WritePassHeader(h, extension_bits, &pos, storage.data()));
  EXPECT_EQ(total_bits, pos);
}

// Ensures Read(Write()) returns the same fields.
TEST(HeadersTest, TestRoundTrip) {
  FrameHeader h;
  h.has_alpha = true;
  h.extensions = 0x800;

  size_t extension_bits, total_bits;
  ASSERT_TRUE(CanEncode(h, &extension_bits, &total_bits));
  EXPECT_EQ(0, extension_bits);
  PaddedBytes storage(DivCeil(total_bits, kBitsPerByte));
  size_t pos = 0;
  ASSERT_TRUE(WritePassHeader(h, extension_bits, &pos, storage.data()));
  EXPECT_EQ(total_bits, pos);

  FrameHeader h2;
  BitReader reader(storage.data(), storage.size());
  ASSERT_TRUE(ReadPassHeader(&reader, &h2));
  EXPECT_EQ(total_bits, reader.BitsRead());

  EXPECT_EQ(h.extensions, h2.extensions);
  EXPECT_EQ(h.has_alpha, h2.has_alpha);
}

#ifndef PIK_CRASH_ON_ERROR
// Ensure out-of-bounds values cause an error.
TEST(HeadersTest, TestOutOfRange) {
  FrameHeader h;
  h.encoding = static_cast<pik::ImageEncoding>(999);
  size_t extension_bits, total_bits;
  ASSERT_FALSE(CanEncode(h, &extension_bits, &total_bits));
}
#endif

struct OldBundle {
  OldBundle() { Bundle::Init(this); }
  constexpr const char* Name() const { return "OldBundle"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    visitor->U32(0x04030281, 1, &old_small);
    visitor->U32(0x20100C07, 0, &old_large);

    visitor->BeginExtensions(&extensions);
    return visitor->EndExtensions();
  }

  uint32_t old_small;
  uint32_t old_large;
  uint64_t extensions;
};

struct NewBundle {
  NewBundle() { Bundle::Init(this); }
  constexpr const char* Name() const { return "NewBundle"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    visitor->U32(0x04030281, 1, &old_small);
    visitor->U32(0x20100807, 0, &old_large);

    visitor->BeginExtensions(&extensions);
    if (extensions & 1) {
      visitor->U32(0x04030282, 2, &new_small);
      visitor->U32(0x20100C09, 0, &new_large);
    }
    return visitor->EndExtensions();
  }

  uint32_t old_small;
  uint32_t old_large;
  uint64_t extensions;

  // If extensions & 1
  uint32_t new_small = 2;
  uint32_t new_large = 0;
};

TEST(HeadersTest, TestNewDecoderOldData) {
  OldBundle old_bundle;
  old_bundle.old_large = 123;
  old_bundle.extensions = 0;

  // Write to bit stream
  size_t pos = 0;
  uint8_t storage[999] = {0};
  size_t extension_bits, total_bits;
  ASSERT_TRUE(Bundle::CanEncode(old_bundle, &extension_bits, &total_bits));
  EXPECT_EQ(0, extension_bits);
  ASSERT_LE(DivCeil(total_bits, kBitsPerByte), sizeof(storage));
  ASSERT_TRUE(Bundle::Write(old_bundle, extension_bits, &pos, storage));
  const size_t bits_written = pos;

  WriteBits(20, 0xA55A, &pos, storage);  // sentinel
  BitReader reader(storage, sizeof(storage));
  NewBundle new_bundle;
  ASSERT_TRUE(Bundle::Read(&reader, &new_bundle));
  EXPECT_EQ(reader.BitsRead(), bits_written);
  EXPECT_EQ(reader.ReadBits(20), 0xA55A);

  // Old fields are the same in both
  EXPECT_EQ(old_bundle.extensions, new_bundle.extensions);
  EXPECT_EQ(old_bundle.old_small, new_bundle.old_small);
  EXPECT_EQ(old_bundle.old_large, new_bundle.old_large);
  // New fields match their defaults
  EXPECT_EQ(2, new_bundle.new_small);
  EXPECT_EQ(0, new_bundle.new_large);
}

TEST(HeadersTest, TestOldDecoderNewData) {
  NewBundle new_bundle;
  new_bundle.old_large = 123;
  new_bundle.extensions = 1;
  new_bundle.new_large = 456;

  // Write to bit stream
  size_t pos = 0;
  uint8_t storage[999] = {0};
  size_t extension_bits, total_bits;
  ASSERT_TRUE(Bundle::CanEncode(new_bundle, &extension_bits, &total_bits));
  EXPECT_NE(0, extension_bits);
  ASSERT_LE(DivCeil(total_bits, kBitsPerByte), sizeof(storage));
  ASSERT_TRUE(Bundle::Write(new_bundle, extension_bits, &pos, storage));
  const size_t bits_written = pos;

  // Ensure Read skips the additional fields
  WriteBits(20, 0xA55A, &pos, storage);  // sentinel
  BitReader reader(storage, sizeof(storage));
  OldBundle old_bundle;
  ASSERT_TRUE(Bundle::Read(&reader, &old_bundle));
  EXPECT_EQ(reader.BitsRead(), bits_written);
  EXPECT_EQ(reader.ReadBits(20), 0xA55A);

  // Old fields are the same in both
  EXPECT_EQ(new_bundle.extensions, old_bundle.extensions);
  EXPECT_EQ(new_bundle.old_small, old_bundle.old_small);
  EXPECT_EQ(new_bundle.old_large, old_bundle.old_large);
  // (Can't check new fields because old decoder doesn't know about them)
}

}  // namespace
}  // namespace pik
