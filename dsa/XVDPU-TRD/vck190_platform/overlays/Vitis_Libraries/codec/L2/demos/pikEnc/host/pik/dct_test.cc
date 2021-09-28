// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/dct.h"

#include <math.h>
#include <string.h>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "pik/block.h"
#include "pik/image.h"

namespace pik {
namespace {

namespace slow_dct {
static double alpha(int u) { return u == 0 ? 0.7071067811865475 : 1.0; }
template <size_t N>
void DCTSlow(double block[N * N]) {
  constexpr size_t block_size = N * N;
  double g[block_size];
  memcpy(g, block, block_size * sizeof(g[0]));
  const float scale = std::sqrt(2.0 / N);
  for (int v = 0; v < N; ++v) {
    for (int u = 0; u < N; ++u) {
      double val = 0.0;
      for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
          val += (alpha(u) * cos((x + 0.5) * u * Pi(1.0 / N)) * alpha(v) *
                  cos((y + 0.5) * v * Pi(1.0 / N)) * g[N * y + x]);
        }
      }
      block[N * v + u] = val * scale * scale;
    }
  }
}

template <size_t N>
void IDCTSlow(double block[N * N]) {
  constexpr size_t block_size = N * N;
  double F[block_size];
  memcpy(F, block, block_size * sizeof(F[0]));
  const float scale = std::sqrt(2.0 / N);
  for (int v = 0; v < N; ++v) {
    for (int u = 0; u < N; ++u) {
      double val = 0.0;
      for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
          val += (alpha(x) * cos(x * (u + 0.5) * Pi(1.0 / N)) * alpha(y) *
                  cos(y * (v + 0.5) * Pi(1.0 / N)) * F[N * y + x]);
        }
      }
      block[N * v + u] = val * scale * scale;
    }
  }
}

// These functions should be equivalent to ComputeTransposedScaledDCT in the pik
// namespace (but slower and implemented for more transform sizes).
template <size_t N, class From, class To>
SIMD_ATTR static void ComputeTransposedScaledDCT(const From& from,
                                                 const To& to) {
  double blockd[N * N] = {};
  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      blockd[y * N + x] = from.Read(y, x);
    }
  }
  DCTSlow<N>(blockd);

  // Scale.
  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      to.Write(
          blockd[N * x + y] * N * N * IDCTScales<N>()[x] * IDCTScales<N>()[y],
          y, x);
    }
  }
}

template <size_t N, class From, class To>
SIMD_ATTR static void ComputeTransposedScaledIDCT(const From& from,
                                                  const To& to) {
  // Scale.
  double blockd[N * N] = {};
  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      blockd[x * N + y] =
          from.Read(y, x) * N * N * DCTScales<N>()[x] * DCTScales<N>()[y];
    }
  }
  IDCTSlow<N>(blockd);

  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      to.Write(blockd[N * y + x], y, x);
    }
  }
}
}  // namespace slow_dct

using slow_dct::DCTSlow;
using slow_dct::IDCTSlow;

// Computes the in-place NxN DCT of block.
// Requires that block is SIMD_ALIGN'ed.
//
// Performs ComputeTransposedScaledDCT and then transposes and scales it to
// obtain "vanilla" DCT.
template <size_t N>
SIMD_ATTR void ComputeDCT(float block[N * N]) {
  ComputeTransposedScaledDCT<N>()(FromBlock<N>(block), ToBlock<N>(block));

  // Untranspose.
  GenericTransposeBlockInplace<N>(FromBlock<N>(block), ToBlock<N>(block));

  // Unscale.
  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      block[N * y + x] *= DCTScales<N>()[x] * DCTScales<N>()[y];
    }
  }
}

// Computes the in-place 8x8 iDCT of block.
// Requires that block is SIMD_ALIGN'ed.
template <int N>
SIMD_ATTR void ComputeIDCT(float block[N * N]) {
  // Unscale.
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      block[N * y + x] *= IDCTScales<N>()[x] * IDCTScales<N>()[y];
    }
  }

  // Untranspose.
  GenericTransposeBlockInplace<N>(FromBlock<N>(block), ToBlock<N>(block));

  ComputeTransposedScaledIDCT<N>()(FromBlock<N>(block), ToBlock<N>(block));
}

template <size_t N, class From, class To>
SIMD_ATTR PIK_INLINE void TransposeBlock(const From& from, const To& to) {
  if (N == 8) {
    TransposeBlock8(from, to);
  } else {
    TransposeBlock16(from, to);
  }
}

template <size_t N, class From, class To>
SIMD_ATTR PIK_INLINE void ColumnDct(const From& from, const To& to) {
  if (N == 8) {
    ColumnDCT8(from, to);
  } else {
    ColumnDCT16(from, to);
  }
}

template <size_t N, class From, class To>
SIMD_ATTR PIK_INLINE void ColumnIdct(const From& from, const To& to) {
  if (N == 8) {
    ColumnIDCT8(from, to);
  } else {
    ColumnIDCT16(from, to);
  }
}

template <size_t N>
SIMD_ATTR void TransposeTest(float accuracy) {
  constexpr size_t block_size = N * N;
  SIMD_ALIGN float src[block_size];
  ToBlock<N> to_src(src);
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      to_src.Write(y * N + x, y, x);
    }
  }
  SIMD_ALIGN float dst[block_size];
  TransposeBlock<N>(FromBlock<N>(src), ToBlock<N>(dst));
  FromBlock<N> from_dst(dst);
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      float expected = x * N + y;
      float actual = from_dst.Read(y, x);
      EXPECT_NEAR(expected, actual, accuracy) << "x = " << x << " y = " << y;
    }
  }
}

TEST(TransposeTest, Transpose8) { TransposeTest<8>(1e-7); }

TEST(TransposeTest, Transpose16) { TransposeTest<16>(1e-7); }

template <size_t N>
SIMD_ATTR void TestDctAccuracy(float accuracy) {
  constexpr size_t block_size = N * N;
  for (int i = 0; i < block_size; ++i) {
    SIMD_ALIGN float fast[block_size] = {0.0f};
    double slow[block_size] = {0.0};
    fast[i] = 1.0;
    slow[i] = 1.0;
    DCTSlow<N>(slow);
    ComputeDCT<N>(fast);
    for (int k = 0; k < block_size; ++k) {
      EXPECT_NEAR(fast[k], slow[k], accuracy) << "i = " << i << " k = " << k;
    }
  }
}

TEST(DctTest, Accuracy8) { TestDctAccuracy<8>(1.1e-7); }

TEST(DctTest, Accuracy16) { TestDctAccuracy<16>(1.1e-7); }

template <size_t N>
SIMD_ATTR void TestIdctAccuracy(float accuracy) {
  constexpr size_t block_size = N * N;
  for (int i = 0; i < block_size; ++i) {
    SIMD_ALIGN float fast[block_size] = {0.0f};
    double slow[block_size] = {0.0};
    fast[i] = 1.0;
    slow[i] = 1.0;
    IDCTSlow<N>(slow);
    ComputeIDCT<N>(fast);
    for (int k = 0; k < block_size; ++k) {
      EXPECT_NEAR(fast[k], slow[k], accuracy) << "i = " << i << " k = " << k;
    }
  }
}

TEST(IdctTest, Accuracy8) { TestIdctAccuracy<8>(1e-7); }

TEST(IdctTest, Accuracy16) { TestIdctAccuracy<16>(1e-7); }

template <size_t N>
SIMD_ATTR void TestInverse(float accuracy) {
  constexpr size_t block_size = N * N;
  for (int i = 0; i < block_size; ++i) {
    SIMD_ALIGN float x[block_size] = {0.0f};
    x[i] = 1.0;

    ComputeIDCT<N>(x);
    ComputeDCT<N>(x);

    for (int k = 0; k < block_size; ++k) {
      EXPECT_NEAR(x[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << " k = " << k;
    }
  }
}

TEST(IdctTest, Inverse8) { TestInverse<8>(1e-6f); }

TEST(IdctTest, Inverse16) { TestInverse<16>(1e-6f); }

template <size_t N>
SIMD_ATTR void TestIdctOrthonormal(float accuracy) {
  constexpr size_t block_size = N * N;
  ImageF xs(block_size, block_size);
  for (int i = 0; i < block_size; ++i) {
    float* x = xs.Row(i);
    for (int j = 0; j < block_size; ++j) x[j] = (i == j) ? 1.0f : 0.0f;
    ComputeIDCT<N>(x);
  }
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      float product = 0.0f;
      for (int k = 0; k < block_size; ++k) {
        product += xs.Row(i)[k] * xs.Row(j)[k];
      }
      EXPECT_NEAR(product, (i == j) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << " j = " << j;
    }
  }
}

TEST(IdctTest, IDCTOrthonormal8) { TestIdctOrthonormal<8>(1e-6f); }

TEST(IdctTest, IDCTOrthonormal16) { TestIdctOrthonormal<16>(1.2e-6f); }

template <size_t N>
SIMD_ATTR void TestDctTranspose(float accuracy) {
  constexpr size_t block_size = N * N;
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      // We check that <e_i, Me_j> = <M^\dagger{}e_i, e_j>.
      // That means (Me_j)_i = (M^\dagger{}e_i)_j

      // x := Me_j
      SIMD_ALIGN float x[block_size] = {0.0f};
      x[j] = 1.0;
      ComputeIDCT<N>(x);
      // y := M^\dagger{}e_i
      SIMD_ALIGN float y[block_size] = {0.0f};
      y[i] = 1.0;
      ComputeDCT<N>(y);

      EXPECT_NEAR(x[i], y[j], accuracy) << "i = " << i << " j = " << j;
    }
  }
}

TEST(IdctTest, Transpose8) { TestDctTranspose<8>(1e-6); }

TEST(IdctTest, Transpose16) { TestDctTranspose<16>(1e-6); }

template <size_t N>
SIMD_ATTR void TestDctOrthonormal(float accuracy) {
  constexpr size_t block_size = N * N;
  ImageF xs(block_size, block_size);
  for (int i = 0; i < block_size; ++i) {
    float* x = xs.Row(i);
    for (int j = 0; j < block_size; ++j) x[j] = (i == j) ? 1.0f : 0.0f;
    ComputeDCT<N>(x);
  }
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      float product = 0.0f;
      for (int k = 0; k < block_size; ++k) {
        product += xs.Row(i)[k] * xs.Row(j)[k];
      }
      EXPECT_NEAR(product, (i == j) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << " j = " << j;
    }
  }
}

TEST(IdctTest, DCTOrthonormal8) { TestDctOrthonormal<8>(1e-6f); }

TEST(IdctTest, DCTOrthonormal16) { TestDctOrthonormal<16>(1e-6f); }

template <size_t N>
SIMD_ATTR void ColumnDctRoundtrip(float accuracy) {
  constexpr size_t block_size = N * N;
  // Though we are only interested in single column result, dct.h has built-in
  // limit on minimal number of columns processed. So, to be safe, we do
  // regular 8x8 block tranformation. On the bright side - we could check all
  // 8 basis vectors at once.
  SIMD_ALIGN float block[block_size];
  ToBlock<N> to(block);
  FromBlock<N> from(block);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      to.Write((i == j) ? 1.0f : 0.0f, i, j);
    }
  }

  ColumnDct<N>(from, to);
  ColumnIdct<N>(from, to);
  constexpr float scale = 1.0f / N;

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      float actual = from.Read(i, j) * scale;
      EXPECT_NEAR(expected, actual, accuracy) << " i=" << i << " j=" << j;
    }
  }
}

TEST(IdctTest, ColumnDctRoundtrip8) { ColumnDctRoundtrip<8>(1e-6f); }

TEST(IdctTest, ColumnDctRoundtrip16) { ColumnDctRoundtrip<16>(1e-6f); }

// See "Steerable Discrete Cosine Transform", Fracastoro G., Fosson S., Magli
// E., https://arxiv.org/pdf/1610.09152.pdf
template <int N>
void RotateDCT(float angle, float block[N * N]) {
  float a2a = std::cos(angle);
  float a2b = -std::sin(angle);
  float b2a = std::sin(angle);
  float b2b = std::cos(angle);
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < y; x++) {
      if (x >= 2 || y >= 2) continue;
      float a = block[N * y + x];
      float b = block[N * x + y];
      block[N * y + x] = a2a * a + b2a * b;
      block[N * x + y] = a2b * a + b2b * b;
    }
  }
}

TEST(RotateTest, ZeroIdentity) {
  for (int i = 0; i < 64; i++) {
    float x[64] = {0.0f};
    x[i] = 1.0;
    RotateDCT<8>(0.0f, x);
    for (int j = 0; j < 64; j++) {
      EXPECT_NEAR(x[j], (i == j) ? 1.0f : 0.0f, 1e-6f);
    }
  }
}

TEST(RotateTest, InverseRotation) {
  const float angle = 0.1f;
  for (int i = 0; i < 64; i++) {
    float x[64] = {0.0f};
    x[i] = 1.0;
    RotateDCT<8>(angle, x);
    RotateDCT<8>(-angle, x);
    for (int j = 0; j < 64; j++) {
      EXPECT_NEAR(x[j], (i == j) ? 1.0f : 0.0f, 1e-6f);
    }
  }
}

template <size_t N>
SIMD_ATTR void TestSlowIsSameDCT(float accuracy, int start = 0,
                                 int end = N * N) {
  for (size_t i = start; i < end; i++) {
    SIMD_ALIGN float block1[N * N] = {};
    SIMD_ALIGN float block2[N * N] = {};
    block1[i] = 1.0;
    block2[i] = 1.0;
    ComputeTransposedScaledDCT<N>()(FromBlock<N>(block1), ToBlock<N>(block1));
    slow_dct::ComputeTransposedScaledDCT<N>(FromBlock<N>(block2),
                                            ToBlock<N>(block2));
    for (int j = 0; j < N * N; j++) {
      EXPECT_NEAR(block1[j], block2[j], accuracy);
    }
  }
}

template <size_t N>
SIMD_ATTR void TestSlowIsSameDCTShard(float accuracy, int shard) {
  TestSlowIsSameDCT<N>(accuracy, N * shard, N * (shard + 1));
}

TEST(SlowDctTest, DCTIsSame2) { TestSlowIsSameDCT<2>(1e-5); }
TEST(SlowDctTest, DCTIsSame4) { TestSlowIsSameDCT<4>(1e-5); }
TEST(SlowDctTest, DCTIsSame8) { TestSlowIsSameDCT<8>(1e-5); }
TEST(SlowDctTest, DCTIsSame16) { TestSlowIsSameDCT<16>(2e-5); }
TEST(SlowDctTest, DCTIsSame32s0) { TestSlowIsSameDCTShard<32>(1e-4, 0); }
TEST(SlowDctTest, DCTIsSame32s1) { TestSlowIsSameDCTShard<32>(1e-4, 1); }
TEST(SlowDctTest, DCTIsSame32s2) { TestSlowIsSameDCTShard<32>(1e-4, 2); }
TEST(SlowDctTest, DCTIsSame32s3) { TestSlowIsSameDCTShard<32>(1e-4, 3); }
TEST(SlowDctTest, DCTIsSame32s4) { TestSlowIsSameDCTShard<32>(1e-4, 4); }
TEST(SlowDctTest, DCTIsSame32s5) { TestSlowIsSameDCTShard<32>(1e-4, 5); }
TEST(SlowDctTest, DCTIsSame32s6) { TestSlowIsSameDCTShard<32>(1e-4, 6); }
TEST(SlowDctTest, DCTIsSame32s7) { TestSlowIsSameDCTShard<32>(1e-4, 7); }
TEST(SlowDctTest, DCTIsSame32s8) { TestSlowIsSameDCTShard<32>(1e-4, 8); }
TEST(SlowDctTest, DCTIsSame32s9) { TestSlowIsSameDCTShard<32>(1e-4, 9); }
TEST(SlowDctTest, DCTIsSame32s10) { TestSlowIsSameDCTShard<32>(1e-4, 10); }
TEST(SlowDctTest, DCTIsSame32s11) { TestSlowIsSameDCTShard<32>(1e-4, 11); }
TEST(SlowDctTest, DCTIsSame32s12) { TestSlowIsSameDCTShard<32>(1e-4, 12); }
TEST(SlowDctTest, DCTIsSame32s13) { TestSlowIsSameDCTShard<32>(1e-4, 13); }
TEST(SlowDctTest, DCTIsSame32s14) { TestSlowIsSameDCTShard<32>(1e-4, 14); }
TEST(SlowDctTest, DCTIsSame32s15) { TestSlowIsSameDCTShard<32>(1e-4, 15); }
TEST(SlowDctTest, DCTIsSame32s16) { TestSlowIsSameDCTShard<32>(1e-4, 16); }
TEST(SlowDctTest, DCTIsSame32s17) { TestSlowIsSameDCTShard<32>(1e-4, 17); }
TEST(SlowDctTest, DCTIsSame32s18) { TestSlowIsSameDCTShard<32>(1e-4, 18); }
TEST(SlowDctTest, DCTIsSame32s19) { TestSlowIsSameDCTShard<32>(1e-4, 19); }
TEST(SlowDctTest, DCTIsSame32s20) { TestSlowIsSameDCTShard<32>(1e-4, 20); }
TEST(SlowDctTest, DCTIsSame32s21) { TestSlowIsSameDCTShard<32>(1e-4, 21); }
TEST(SlowDctTest, DCTIsSame32s22) { TestSlowIsSameDCTShard<32>(1e-4, 22); }
TEST(SlowDctTest, DCTIsSame32s23) { TestSlowIsSameDCTShard<32>(1e-4, 23); }
TEST(SlowDctTest, DCTIsSame32s24) { TestSlowIsSameDCTShard<32>(1e-4, 24); }
TEST(SlowDctTest, DCTIsSame32s25) { TestSlowIsSameDCTShard<32>(1e-4, 25); }
TEST(SlowDctTest, DCTIsSame32s26) { TestSlowIsSameDCTShard<32>(1e-4, 26); }
TEST(SlowDctTest, DCTIsSame32s27) { TestSlowIsSameDCTShard<32>(1e-4, 27); }
TEST(SlowDctTest, DCTIsSame32s28) { TestSlowIsSameDCTShard<32>(1e-4, 28); }
TEST(SlowDctTest, DCTIsSame32s29) { TestSlowIsSameDCTShard<32>(1e-4, 29); }
TEST(SlowDctTest, DCTIsSame32s30) { TestSlowIsSameDCTShard<32>(1e-4, 30); }
TEST(SlowDctTest, DCTIsSame32s31) { TestSlowIsSameDCTShard<32>(1e-4, 31); }

template <size_t N>
SIMD_ATTR void TestSlowIsSameIDCT(float accuracy, int start = 0,
                                  int end = N * N) {
  for (size_t i = start; i < end; i++) {
    SIMD_ALIGN float block1[N * N] = {};
    SIMD_ALIGN float block2[N * N] = {};
    block1[i] = 1.0;
    block2[i] = 1.0;
    ComputeTransposedScaledIDCT<N>()(FromBlock<N>(block1), ToBlock<N>(block1));
    slow_dct::ComputeTransposedScaledIDCT<N>(FromBlock<N>(block2),
                                             ToBlock<N>(block2));
    for (int j = 0; j < N * N; j++) {
      EXPECT_NEAR(block1[j], block2[j], accuracy);
    }
  }
}

template <size_t N>
SIMD_ATTR void TestSlowIsSameIDCTShard(float accuracy, int shard) {
  TestSlowIsSameIDCT<N>(accuracy, N * shard, N * (shard + 1));
}

TEST(SlowDctTest, IDCTIsSame2) { TestSlowIsSameIDCT<2>(1e-5); }
TEST(SlowDctTest, IDCTIsSame4) { TestSlowIsSameIDCT<4>(1e-5); }
TEST(SlowDctTest, IDCTIsSame8) { TestSlowIsSameIDCT<8>(1e-5); }
TEST(SlowDctTest, IDCTIsSame16) { TestSlowIsSameIDCT<16>(2e-5); }
TEST(SlowDctTest, IDCTIsSame32s0) { TestSlowIsSameIDCTShard<32>(1e-4, 0); }
TEST(SlowDctTest, IDCTIsSame32s1) { TestSlowIsSameIDCTShard<32>(1e-4, 1); }
TEST(SlowDctTest, IDCTIsSame32s2) { TestSlowIsSameIDCTShard<32>(1e-4, 2); }
TEST(SlowDctTest, IDCTIsSame32s3) { TestSlowIsSameIDCTShard<32>(1e-4, 3); }
TEST(SlowDctTest, IDCTIsSame32s4) { TestSlowIsSameIDCTShard<32>(1e-4, 4); }
TEST(SlowDctTest, IDCTIsSame32s5) { TestSlowIsSameIDCTShard<32>(1e-4, 5); }
TEST(SlowDctTest, IDCTIsSame32s6) { TestSlowIsSameIDCTShard<32>(1e-4, 6); }
TEST(SlowDctTest, IDCTIsSame32s7) { TestSlowIsSameIDCTShard<32>(1e-4, 7); }
TEST(SlowDctTest, IDCTIsSame32s8) { TestSlowIsSameIDCTShard<32>(1e-4, 8); }
TEST(SlowDctTest, IDCTIsSame32s9) { TestSlowIsSameIDCTShard<32>(1e-4, 9); }
TEST(SlowDctTest, IDCTIsSame32s10) { TestSlowIsSameIDCTShard<32>(1e-4, 10); }
TEST(SlowDctTest, IDCTIsSame32s11) { TestSlowIsSameIDCTShard<32>(1e-4, 11); }
TEST(SlowDctTest, IDCTIsSame32s12) { TestSlowIsSameIDCTShard<32>(1e-4, 12); }
TEST(SlowDctTest, IDCTIsSame32s13) { TestSlowIsSameIDCTShard<32>(1e-4, 13); }
TEST(SlowDctTest, IDCTIsSame32s14) { TestSlowIsSameIDCTShard<32>(1e-4, 14); }
TEST(SlowDctTest, IDCTIsSame32s15) { TestSlowIsSameIDCTShard<32>(1e-4, 15); }
TEST(SlowDctTest, IDCTIsSame32s16) { TestSlowIsSameIDCTShard<32>(1e-4, 16); }
TEST(SlowDctTest, IDCTIsSame32s17) { TestSlowIsSameIDCTShard<32>(1e-4, 17); }
TEST(SlowDctTest, IDCTIsSame32s18) { TestSlowIsSameIDCTShard<32>(1e-4, 18); }
TEST(SlowDctTest, IDCTIsSame32s19) { TestSlowIsSameIDCTShard<32>(1e-4, 19); }
TEST(SlowDctTest, IDCTIsSame32s20) { TestSlowIsSameIDCTShard<32>(1e-4, 20); }
TEST(SlowDctTest, IDCTIsSame32s21) { TestSlowIsSameIDCTShard<32>(1e-4, 21); }
TEST(SlowDctTest, IDCTIsSame32s22) { TestSlowIsSameIDCTShard<32>(1e-4, 22); }
TEST(SlowDctTest, IDCTIsSame32s23) { TestSlowIsSameIDCTShard<32>(1e-4, 23); }
TEST(SlowDctTest, IDCTIsSame32s24) { TestSlowIsSameIDCTShard<32>(1e-4, 24); }
TEST(SlowDctTest, IDCTIsSame32s25) { TestSlowIsSameIDCTShard<32>(1e-4, 25); }
TEST(SlowDctTest, IDCTIsSame32s26) { TestSlowIsSameIDCTShard<32>(1e-4, 26); }
TEST(SlowDctTest, IDCTIsSame32s27) { TestSlowIsSameIDCTShard<32>(1e-4, 27); }
TEST(SlowDctTest, IDCTIsSame32s28) { TestSlowIsSameIDCTShard<32>(1e-4, 28); }
TEST(SlowDctTest, IDCTIsSame32s29) { TestSlowIsSameIDCTShard<32>(1e-4, 29); }
TEST(SlowDctTest, IDCTIsSame32s30) { TestSlowIsSameIDCTShard<32>(1e-4, 30); }
TEST(SlowDctTest, IDCTIsSame32s31) { TestSlowIsSameIDCTShard<32>(1e-4, 31); }

template <size_t N>
SIMD_ATTR void TestSlowInverse(float accuracy) {
  constexpr size_t block_size = N * N;
  for (int i = 0; i < block_size; ++i) {
    float x[block_size] = {0.0f};
    x[i] = 1.0;

    slow_dct::ComputeTransposedScaledDCT<N>(FromBlock<N>(x),
                                            ScaleToBlock<N>(x));
    slow_dct::ComputeTransposedScaledIDCT<N>(FromBlock<N>(x), ToBlock<N>(x));

    for (int k = 0; k < block_size; ++k) {
      EXPECT_NEAR(x[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << " k = " << k;
    }
  }
}

TEST(SlowDctTest, SlowInverse2) { TestSlowInverse<2>(1e-5); }
TEST(SlowDctTest, SlowInverse4) { TestSlowInverse<4>(1e-5); }
TEST(SlowDctTest, SlowInverse8) { TestSlowInverse<8>(1e-5); }
TEST(SlowDctTest, SlowInverse16) { TestSlowInverse<16>(1e-5); }

}  // namespace
}  // namespace pik
