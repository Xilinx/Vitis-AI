// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/bit_reader.h"

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/padded_bytes.h"
#include "pik/write_bits.h"

namespace pik {
namespace {

TEST(BitReaderTest, ExtendsWithZeroes) {
  constexpr int kSize = 21;
  std::vector<uint8_t> data(kSize + 4, 0xff);
  for (int n_bytes = 0; n_bytes < kSize; n_bytes++) {
    BitReader br(data.data(), n_bytes);
    for (int i = 0; i < n_bytes * 8; i++) {
      ASSERT_EQ(br.ReadBits(1), 1) << "n_bytes=" << n_bytes << " i=" << i;
    }
    for (int i = 0; i < (kSize - n_bytes) * 8; i++) {
      ASSERT_EQ(br.ReadBits(1), 0) << "n_bytes=" << n_bytes << " i=" << i;
    }
  }
}

struct Symbol {
  uint32_t num_bits;
  uint32_t value;
};

// Reading from WriteBits output gives the same values.
TEST(BitReaderTest, TestRoundTrip) {
  ThreadPool pool(8);
  pool.Run(0, 1000, [](const int task, const int thread) {
    constexpr size_t kMaxBits = 8000;
    PaddedBytes storage(DivCeil(kMaxBits, kBitsPerByte));

    std::vector<Symbol> symbols;
    symbols.reserve(1000);

    std::mt19937 rng(55537 + 129 * task);
    std::uniform_int_distribution<> dist(1, 32);  // closed interval

    uint64_t pos = 0;
    for (;;) {
      const uint32_t num_bits = dist(rng);
      if (pos + num_bits > kMaxBits) break;
      const uint32_t value = rng() >> (32 - num_bits);
      symbols.push_back({num_bits, value});
      WriteBits(num_bits, value, &pos, storage.data());
    }

    BitReader reader(storage.data(), DivCeil(pos, kBitsPerByte));
    for (const Symbol& s : symbols) {
      EXPECT_EQ(s.value, reader.ReadBits(s.num_bits));
    }
  });
}

// SkipBits is the same as reading that many bits.
TEST(BitReaderTest, TestSkip) {
  ThreadPool pool(8);
  pool.Run(0, 96, [](const int task, const int thread) {
    constexpr size_t kSize = 100;
    PaddedBytes storage(kSize);

    for (size_t skip = 0; skip < 128; ++skip) {
      memset(storage.data(), 0, kSize);
      size_t pos = 0;
      // Start with "task" 1-bits.
      for (int i = 0; i < task; ++i) {
        WriteBits(1, 1, &pos, storage.data());
      }

      // Write 0-bits that we will skip over
      for (size_t i = 0; i < skip; ++i) {
        WriteBits(1, 0, &pos, storage.data());
      }

      // Write terminator bits '101'
      WriteBits(3, 5, &pos, storage.data());
      EXPECT_EQ(task + skip + 3, pos);
      EXPECT_LT(pos, kSize * 8);

      BitReader reader1(storage.data(), kSize);
      BitReader reader2(storage.data(), kSize);
      // Verify initial 1-bits
      for (int i = 0; i < task; ++i) {
        EXPECT_EQ(1, reader1.ReadBits(1));
        EXPECT_EQ(1, reader2.ReadBits(1));
      }

      // SkipBits or manually read "skip" bits
      reader1.SkipBits(skip);
      for (size_t i = 0; i < skip; ++i) {
        EXPECT_EQ(0, reader2.ReadBits(1)) << " skip=" << skip << " i=" << i;
      }
      EXPECT_EQ(reader1.BitsRead(), reader2.BitsRead());

      // Ensure both readers see the terminator bits.
      EXPECT_EQ(5, reader1.ReadBits(3));
      EXPECT_EQ(5, reader2.ReadBits(3));
    }
  });
}

}  // namespace
}  // namespace pik
