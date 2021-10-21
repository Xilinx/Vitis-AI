// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/entropy_coder.h"

#include <stdint.h>
#include <stdio.h>
#include <array>
#include <cmath>
#include <random>
#include <utility>

#include "gtest/gtest.h"
#include "pik/ans_decode.h"
#include "pik/common.h"
#include "pik/compressed_image.h"
#include "pik/data_parallel.h"
#include "pik/dct_util.h"
#include "pik/image_test_utils.h"
#include "pik/opsin_image.h"
#include "pik/quantizer.h"
#include "pik/single_image_handler.h"

namespace pik {
namespace {

template <typename Random>
class UniformWithPeakAtZero {
 public:
  explicit UniformWithPeakAtZero(Random* rng) : rng_(*rng) {}
  int16_t operator()() const {
    std::uniform_int_distribution<> dist(-4096, 4096);
    int16_t val = dist(rng_);
    return val < -3072 ? val + 3072 : val > 3072 ? val - 3072 : 0;
  }

 private:
  Random& rng_;
};

template <typename Random>
class Uniform {
 public:
  Uniform(Random* rng, int min, int max)  // inclusive
      : min_(min), max_(max), rng_(*rng) {}
  int16_t operator()(int, size_t, size_t) const {
    std::uniform_int_distribution<int> dist(min_, max_);
    return dist(rng_);
  }

 private:
  const int min_;
  const int max_;
  Random& rng_;
};

template <typename Random>
class BinaryDistribution {
 public:
  explicit BinaryDistribution(Random* rng, float prob)
      : prob_(prob), rng_(*rng) {}
  int operator()() const {
    std::binomial_distribution<int> dist(1, prob_);
    return dist(rng_);
  }

 private:
  const float prob_;
  Random& rng_;
};

TEST(EntropyCoderTest, EncodeDecodeCoeffs) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;

  const size_t ysize = 1 << 5;
  const size_t xsize = 1 << 5;

  AcStrategyImage ac_strategy(xsize, ysize);

  ImageI quant_field(xsize, ysize);
  RandomFillImage(&quant_field, 1, 256, 7777);
  Rect rect(0, 0, xsize, ysize);
  std::vector<Token> tokens;
  TokenizeQuantField(rect, quant_field, nullptr, ac_strategy, &tokens);

  Image3S coeffs_in(xsize * block_size, ysize);
  std::mt19937_64 rng;
  UniformWithPeakAtZero<std::mt19937_64> generator(&rng);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      int16_t* PIK_RESTRICT row = coeffs_in.PlaneRow(c, y);
      for (size_t x = 0; x < xsize * block_size; ++x) {
        row[x] = generator();
      }
    }
  }
  int order[kOrderContexts * block_size];
  ComputeCoeffOrder(coeffs_in, Rect(coeffs_in), order);
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  TokenizeCoefficients(order, rect, coeffs_in, &tokens);

  std::string data = EncodeImageData(rect, DCImage(coeffs_in), nullptr);
  data += EncodeCoeffOrders(order, nullptr);
  data += BuildAndEncodeHistograms(kNumContexts, {tokens}, &codes, &context_map,
                                   nullptr);
  data += WriteTokens(tokens, codes, context_map, nullptr);
  size_t data_size = data.size();
  data.resize(data_size + 4);

  int order_out[kOrderContexts * block_size];
  ImageI quant_field_out(xsize, ysize);
  Image3S dc_out(xsize, ysize);
  const uint8_t* input = reinterpret_cast<const uint8_t*>(data.data());
  BitReader br(input, data_size);
  DecodeImage(&br, rect, &dc_out);
  for (int c = 0; c < kOrderContexts; ++c) {
    DecodeCoeffOrder(&order_out[c * block_size], &br);
  }
  ASSERT_TRUE(br.JumpToByteBoundary());
  ANSCode code;
  std::vector<uint8_t> context_map_out;
  DecodeHistograms(&br, kNumContexts, 256, &code, &context_map_out);
  Image3S ac_out(xsize * block_size, ysize);
  Image3I num_nzeroes(xsize, ysize);
  ANSSymbolReader decoder(&code);

  DecodeQuantField(&br, &decoder, context_map_out, rect, ac_strategy,
                   &quant_field_out, nullptr);

  DecodeAC(context_map_out, order_out, &br, &decoder, &ac_out, rect,
           &num_nzeroes);
  ASSERT_EQ(br.Position(), data_size);

  FillDC(dc_out, &ac_out);
  VerifyEqual(quant_field, quant_field_out);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const int16_t* PIK_RESTRICT row_in = coeffs_in.PlaneRow(c, y);
      const int16_t* PIK_RESTRICT row_out = ac_out.PlaneRow(c, y);  // and DC
      for (size_t x = 0; x < xsize * block_size; ++x) {
        ASSERT_EQ(row_out[x], row_in[x])
            << " c = " << c << " x = " << x << " y = " << y;
      }
    }
  }
}

#ifdef NDEBUG
constexpr size_t kReps = 20;
#else
// Unoptimized builds take too long otherwise.
constexpr size_t kReps = 1;
#endif

template <typename Random>
void RoundtripPrediction(size_t xsize_blocks, size_t ysize_blocks,
                         Random* rng) {
  const Rect tmp_rect(0, 0, xsize_blocks, ysize_blocks);
  Image3S input(xsize_blocks, ysize_blocks);
  Uniform<Random> generator(rng, -32768, 32767);
  Image3S tmp_residuals(xsize_blocks, ysize_blocks);
  ImageS tmp_y(xsize_blocks, ysize_blocks);
  ImageS tmp_xz_residual(xsize_blocks * 2, ysize_blocks);
  ImageS tmp_xz_expanded(xsize_blocks * 2, ysize_blocks);
  for (size_t rep = 0; rep < kReps; ++rep) {
    GenerateImage(generator, &input);
    ShrinkDC(tmp_rect, input, &tmp_residuals);

    Image3S output = CopyImage(tmp_residuals);
    ExpandDC(tmp_rect, &output, &tmp_y, &tmp_xz_residual, &tmp_xz_expanded);
    VerifyRelativeError(input, output, 1E-5, 1E-5);
  }
}

TEST(EntropyCoderTest, PredictRoundtrip) {
  ThreadPool pool(8);

  pool.Run(1, 70, [](const int task, const int thread) {
    std::mt19937_64 rng(129 * task);
    const size_t xsize_blocks = task;
    for (size_t ysize_blocks = 1; ysize_blocks < 70; ysize_blocks++) {
      RoundtripPrediction(xsize_blocks, ysize_blocks, &rng);
    }
  });
}

// Multi-tile Shrink/Expand with tmp buffer to ensure they use rect.
TEST(EntropyCoderTest, PredictRoundtripTiled) {
  const size_t kWidthInTiles = 3;
  const size_t kNumTiles = kWidthInTiles * kWidthInTiles;
  const size_t tile_dim = 64;
  const size_t xsize = kWidthInTiles * tile_dim;
  const size_t ysize = kWidthInTiles * tile_dim;
  Image3S input(xsize, ysize);
  Image3S output(xsize, ysize);
  std::mt19937_64 rng(65537);
  Uniform<std::mt19937_64> generator(&rng, -32768, 32767);

  for (size_t rep = 0; rep < kReps; ++rep) {
    GenerateImage(generator, &input);

    ThreadPool pool(8);
    pool.Run(0, kNumTiles, [&](const int task, const int thread) {
      // Buffers (don't bother reusing/making thread-specific)
      Image3S tmp_residuals(tile_dim, tile_dim);
      ImageS tmp_y(tile_dim, tile_dim);
      ImageS tmp_xz_residual(tile_dim * 2, tile_dim);
      ImageS tmp_xz_expanded(tile_dim * 2, tile_dim);

      const size_t tx = task % kWidthInTiles;
      const size_t ty = task / kWidthInTiles;
      const Rect rect(tx * tile_dim, ty * tile_dim, tile_dim, tile_dim, xsize,
                      ysize);

      ShrinkDC(rect, input, &tmp_residuals);

      // Copy from tmp_residuals into rect within output.
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < rect.ysize(); ++y) {
          const int16_t* PIK_RESTRICT row_from = tmp_residuals.PlaneRow(c, y);
          int16_t* PIK_RESTRICT row_to = rect.PlaneRow(&output, c, y);
          memcpy(row_to, row_from, rect.xsize() * sizeof(*row_to));
        }
      }

      ExpandDC(rect, &output, &tmp_y, &tmp_xz_residual, &tmp_xz_expanded);
    });

    VerifyRelativeError(input, output, 1E-5, 1E-5);
  }
}

TEST(EntropyCoderTest, PackUnpack) {
  for (int32_t i = -31; i < 32; ++i) {
    uint32_t packed = PackSigned(i);
    EXPECT_LT(packed, 63);
    int32_t unpacked = UnpackSigned(packed);
    EXPECT_EQ(i, unpacked);
  }
}

TEST(EntropyCoderTest, EncodeDecodeVarUint) {
  // When n == 0 there is only one, but most important case 0 <-> (0, 0).
  for (int n = 0; n < 6; ++n) {
    uint32_t count = 1 << n;
    uint32_t base = count - 1;
    for (uint32_t i = 0; i < count; ++i) {
      int nbits = -1;
      int bits = -1;
      uint32_t value = base + i;
      EncodeVarLenUint(value, &nbits, &bits);
      EXPECT_EQ(n, nbits);
      EXPECT_EQ(i, bits);
      EXPECT_EQ(value, DecodeVarLenUint(nbits, bits));
    }
  }
}

}  // namespace
}  // namespace pik
