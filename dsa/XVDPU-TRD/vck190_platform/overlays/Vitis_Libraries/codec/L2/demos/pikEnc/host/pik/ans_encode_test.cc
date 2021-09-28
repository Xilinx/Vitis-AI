// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ans_encode.h"

#include "gtest/gtest.h"

namespace pik {
namespace {

void TestANSSymbolInfos(const ANSEncSymbolInfo* info, size_t len) {
  int total = 0;
  for (size_t i = 0; i < len; ++i) {
    total += info[i].freq_;
  }
  EXPECT_EQ(total, ANS_TAB_SIZE);
}

TEST(ANSEncodeTest, EncodeCountsBig) {
  int counts[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7,
  };
  const int kLength = sizeof(counts) / sizeof(counts[0]);
  ANSEncSymbolInfo info[kLength];
  BuildAndStoreANSEncodingData(counts, kLength, info, nullptr, nullptr);
  TestANSSymbolInfos(info, kLength);
}

constexpr int kMaxAlphabetSize = 1024;

void TestBimodalCounts(int n_small, int n_large) {
  if (n_small + n_large > kMaxAlphabetSize || n_small + n_large == 0) return;
  constexpr int kLogTableSize = 10;
  constexpr int kTableSize = 1 << kLogTableSize;
  const float small_p = 0.75 * (1.0 / kTableSize);
  const float large_p = (1.0 - n_small * small_p) / n_large;
  if (large_p < small_p) return;
  std::vector<int> counts(n_small + n_large);
  constexpr int kTotalCounts = 1 << 20;
  for (int i = 0; i < n_small + n_large; i++) {
    counts[i] = kTotalCounts * (i < n_small ? small_p : large_p);
  }
  std::vector<ANSEncSymbolInfo> info(counts.size());
  // Just test that NormalizeCounts() succeeds and this function doesn't crash.
  BuildAndStoreANSEncodingData(&counts[0], counts.size(), &info[0], nullptr,
                               nullptr);
  TestANSSymbolInfos(&info[0], counts.size());
}

TEST(ANSEncodeTest, EncodeCountsBimodal) {
  for (int n_small = 0; n_small < kMaxAlphabetSize; n_small++)
    for (int n_large = 0; n_large < kMaxAlphabetSize; n_large++)
      TestBimodalCounts(n_small, n_large);
}

}  // namespace
}  // namespace pik
