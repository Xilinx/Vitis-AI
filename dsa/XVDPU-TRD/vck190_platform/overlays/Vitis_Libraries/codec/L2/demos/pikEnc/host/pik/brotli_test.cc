// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/brotli.h"

#include <random>

#include "gtest/gtest.h"
#include "pik/data_parallel.h"

namespace pik {
namespace {

TEST(BrotliTest, TestCompressEmpty) {
  PaddedBytes in;
  PaddedBytes out;
  EXPECT_TRUE(BrotliCompress(6, in, &out));
  EXPECT_TRUE(in.empty());
}

TEST(BrotliTest, TestDecompressEmpty) {
  size_t bytes_read = 0;
  PaddedBytes in;
  PaddedBytes out;
  EXPECT_FALSE(BrotliDecompress(in, 1, &bytes_read, &out));
  EXPECT_EQ(0, bytes_read);
  EXPECT_TRUE(in.empty());
  EXPECT_TRUE(out.empty());
}

TEST(BrotliTest, TestRoundTrip) {
  ThreadPool pool(0);
  pool.Run(1, 65, [](const int task, const int thread) {
    const size_t size = task;

    PaddedBytes in(size);
    std::mt19937_64 rng(thread * 65537 + task * 129);
    std::generate(in.begin(), in.end(), rng);
    PaddedBytes compressed;
    PaddedBytes out;

    for (int quality = 1; quality < 7; ++quality) {
      compressed.clear();
      EXPECT_TRUE(BrotliCompress(quality, in, &compressed));
      size_t bytes_read = 0;
      out.clear();
      EXPECT_TRUE(BrotliDecompress(compressed, size, &bytes_read, &out));
      EXPECT_EQ(compressed.size(), bytes_read);
      EXPECT_EQ(in.size(), out.size());
      for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != out[i]) {
          printf("Mismatch at %zu (%zu)\n", i, size);
          exit(1);
        }
      }
    }
  });
}

}  // namespace
}  // namespace pik
