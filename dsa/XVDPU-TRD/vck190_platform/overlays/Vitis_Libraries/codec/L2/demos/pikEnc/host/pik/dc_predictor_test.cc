// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/dc_predictor.h"

#include <random>
#include <stddef.h>
#include <stdlib.h>

#include "gtest/gtest.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/data_parallel.h"
#include "pik/image_test_utils.h"

namespace pik {
namespace {

class GeneratorConstant {
 public:
  explicit GeneratorConstant(const DC dc) : dc_(dc) {}
  DC operator()(const size_t x, const size_t y, const size_t c) const {
    return dc_;
  }

 private:
  const DC dc_;
};

struct GeneratorX {
  DC operator()(const size_t x, const size_t y, const size_t c) const {
    return x;
  }
};

struct GeneratorY {
  DC operator()(const size_t x, const size_t y, const size_t c) const {
    return y;
  }
};

#ifdef NDEBUG
constexpr size_t kReps = 30;
#else
// Unoptimized builds take too long otherwise.
constexpr size_t kReps = 1;
#endif

// Expand(Shrink()) computes the original input coefficients.
template <typename Random>
SIMD_ATTR void RoundTripImage(const size_t xsize, const size_t ysize,
                              Random* rng) {
  // Left/right halves to ensure ShrinkY use their rect arg:
  const size_t xsize_div2 = DivCeil(xsize, size_t(2));
  const Rect rect1(0, 0, xsize_div2, ysize, xsize_div2, ysize);
  const Rect rect2(xsize_div2, 0, xsize / 2, ysize, xsize, ysize);
  Image<DC> in_y(xsize, ysize);
  Image<DC> residuals(xsize, ysize);
  Image<DC> tmp_expanded1(rect1.xsize(), rect1.ysize());
  Image<DC> tmp_expanded2(rect2.xsize(), rect2.ysize());
  GeneratorRandom<DC, Random> generator(rng, -32768, 32767);
  // Realizations of random values
  for (size_t rep = 0; rep < kReps; ++rep) {
    GenerateImage(generator, &in_y);

    ShrinkY(rect1, in_y, rect1, &residuals);
    ExpandY(rect1, residuals, &tmp_expanded1);
    VerifyRelativeError(CopyImage(rect1, in_y), tmp_expanded1, 0, 0);

    if (rect2.xsize() != 0) {
      ShrinkY(rect2, in_y, rect2, &residuals);
      ExpandY(rect2, residuals, &tmp_expanded2);
      VerifyRelativeError(CopyImage(rect2, in_y), tmp_expanded2, 0, 0);
    }
  }
}

template <typename Random>
SIMD_ATTR void RoundTripImageXB(const size_t xsize, const size_t ysize,
                                Random* rng) {
  const Rect tmp_rect(0, 0, xsize, ysize, xsize, ysize);
  Image<DC> tmp_y(xsize, ysize);
  Image<DC> tmp_xb(2 * xsize, ysize);
  Image<DC> tmp_xb_residuals(2 * xsize, ysize);
  Image<DC> tmp_xb_expanded(2 * xsize, ysize);
  GeneratorRandom<DC, Random> generator(rng, -32768, 32767);

  // Realizations of random values
  for (size_t rep = 0; rep < kReps; ++rep) {
    GenerateImage(generator, &tmp_y);
    GenerateImage(generator, &tmp_xb);
    ShrinkXB(tmp_rect, tmp_y, tmp_xb, &tmp_xb_residuals);
    ExpandXB(xsize, ysize, tmp_y, tmp_xb_residuals, &tmp_xb_expanded);
    VerifyEqual(tmp_xb, tmp_xb_expanded);
  }
}

SIMD_ATTR void TestRoundTripImpl() {
  ThreadPool pool(8);
  pool.Run(0, 40, [](const int task, const int thread) {
    std::mt19937_64 rng(task * 65537);
    for (size_t xsize = 1; xsize < 32; ++xsize) {
      for (size_t ysize = 1; ysize < 8; ++ysize) {
        RoundTripImage(xsize, ysize, &rng);
        RoundTripImageXB(xsize, ysize, &rng);
      }
    }
  });
}

TEST(DCPredictorTest, TestRoundTrip) { TestRoundTripImpl(); }

template <class Generator>
SIMD_ATTR uint64_t SumAbs(const size_t xsize, const size_t ysize,
                          const Generator& generator, Image<DC>* dc,
                          Image<DC>* residuals) {
  const Rect rect(0, 0, xsize, ysize, xsize, ysize);
  GenerateImage(generator, dc);
  ShrinkY(rect, *dc, rect, residuals);

  uint64_t sum_abs = 0;
  for (size_t y = 0; y < residuals->ysize(); ++y) {
    const DC* const PIK_RESTRICT row = residuals->Row(y);
    for (size_t x = 0; x < residuals->xsize(); ++x) {
      sum_abs += std::abs(row[x]);
    }
  }
  return sum_abs;
}

SIMD_ATTR void TestSumAbsImpl() {
  for (size_t ysize = 1; ysize < 8; ++ysize) {
    for (size_t xsize = 1; xsize < 32; ++xsize) {
      Image<DC> dc(xsize, ysize);
      Image<DC> residuals(xsize, ysize);
      auto gen_100 = GeneratorConstant(100);
      auto gen_neg_100 = GeneratorConstant(-100);
      auto gen_0 = GeneratorConstant(0);
      ASSERT_EQ(100, SumAbs(xsize, ysize, gen_100, &dc, &residuals));
      ASSERT_EQ(100, SumAbs(xsize, ysize, gen_neg_100, &dc, &residuals));
      ASSERT_EQ(0, SumAbs(xsize, ysize, gen_0, &dc, &residuals));
      const uint64_t sum_x =
          SumAbs(xsize, ysize, GeneratorX(), &dc, &residuals);
      const uint64_t sum_y =
          SumAbs(xsize, ysize, GeneratorY(), &dc, &residuals);
      EXPECT_LT(sum_x, 2 * xsize + ysize + 2);
      EXPECT_LT(sum_y, ysize);
    }
  }
}

TEST(DCPredictorTest, TestSumAbs) { TestSumAbsImpl(); }

template <typename V>
class Predictors {
 public:
  static const size_t kNum = 8;
  struct Y {
    Y() {}

    PIK_INLINE void operator()(const V* const PIK_RESTRICT pos,
                               const intptr_t neg_col_stride,
                               const intptr_t neg_row_stride,
                               V pred[kNum]) const {
      const V w = pos[neg_col_stride];
      const V n = pos[neg_row_stride];
      const V l = pos[neg_row_stride + neg_col_stride];
      const V r = pos[neg_row_stride - neg_col_stride];
      pred[0] = Average(Average(n, w), r);
      pred[1] = Average(w, n);
      pred[2] = Average(n, r);
      pred[3] = Average(w, l);
      pred[4] = Average(l, n);
      pred[5] = w;
      pred[6] = ClampedGradient(n, w, l);
      pred[7] = n;
    }
  };

  struct UV {
    UV() {}

    PIK_INLINE void operator()(const V* const PIK_RESTRICT pos,
                               const intptr_t neg_col_stride,
                               const intptr_t neg_row_stride,
                               V pred[kNum]) const {
      const V w = pos[neg_col_stride];
      const V n = pos[neg_row_stride];
      const V l = pos[neg_row_stride + neg_col_stride];
      const V r = pos[neg_row_stride - neg_col_stride];
      pred[0] = ClampedGradient(n, w, l);
      pred[1] = Average(n, w);
      pred[2] = n;
      pred[3] = Average(n, r);
      pred[4] = w;
      pred[5] = Average(w, l);
      pred[6] = r;
      pred[7] = Average(Average(w, r), n);
    }
  };

 private:
  // Clamps gradient to the min/max of n, w, l.
  static PIK_INLINE V ClampedGradient(const V& n, const V& w, const V& l) {
    const V grad = n + w - l;
    const V min = std::min(n, std::min(w, l));
    const V max = std::max(n, std::max(w, l));
    return std::min(std::max(min, grad), max);
  }

  static PIK_INLINE V Average(const V& v0, const V& v1) {
    return (v0 + v1) >> 1;
  }
};

template <class Predictor, typename V>
V MinCostPrediction(const V* const PIK_RESTRICT dc, size_t x, size_t y,
                    size_t xsize, intptr_t neg_col_stride,
                    intptr_t neg_row_stride) {
  if (y == 0) {
    return x ? dc[neg_col_stride] : 0;
  } else if (x == 0) {
    return dc[neg_row_stride];
  } else if (x == 1 || x + 1 == xsize) {
    return dc[neg_col_stride];
  } else {
    const Predictor predictor;
    V pred[Predictors<V>::kNum];
    predictor(dc, neg_col_stride, neg_row_stride, pred);
    V pred_w[Predictors<V>::kNum];
    predictor(&dc[neg_col_stride], neg_col_stride, neg_row_stride, pred_w);
    const V w = dc[neg_col_stride];
    const V n = dc[neg_row_stride];
    V costs[Predictors<V>::kNum];
    for (int i = 0; i < Predictors<V>::kNum; ++i) {
      costs[i] = std::abs(w - pred_w[i]);
    }
    V pred_n[Predictors<V>::kNum];
    if (y > 1) {
      predictor(&dc[neg_row_stride], neg_col_stride, neg_row_stride, pred_n);
    } else {
      predictor(&dc[neg_row_stride], neg_col_stride, 0, pred_n);
    }
    for (int i = 0; i < Predictors<V>::kNum; ++i) {
      costs[i] += std::abs(n - pred_n[i]);
    }
    const int idx =
        std::min_element(costs, costs + Predictors<V>::kNum) - costs;
    return pred[idx];
  }
}

template <class Predictor, typename V>
V MinCostYPrediction(const V* const PIK_RESTRICT dc, size_t x, size_t y,
                     size_t xsize, intptr_t neg_col_stride,
                     intptr_t neg_row_stride, int uv_predictor) {
  if (y == 0) {
    return x ? dc[neg_col_stride] : 0;
  } else if (x == 0) {
    return dc[neg_row_stride];
  } else if (x == 1 || x + 1 == xsize) {
    return dc[neg_col_stride];
  } else {
    const Predictor predictor;
    V pred[Predictors<V>::kNum];
    predictor(dc, neg_col_stride, neg_row_stride, pred);
    return pred[uv_predictor];
  }
}

template <typename V>
V MinCostPredict(const V* const PIK_RESTRICT dc, size_t x, size_t y,
                 size_t xsize, intptr_t neg_col_stride, intptr_t neg_row_stride,
                 bool use_uv_predictor, int uv_predictor) {
  return use_uv_predictor
             ? MinCostYPrediction<typename Predictors<V>::UV>(
                   dc, x, y, xsize, neg_col_stride, neg_row_stride,
                   uv_predictor)
             : MinCostPrediction<typename Predictors<V>::Y>(
                   dc, x, y, xsize, neg_col_stride, neg_row_stride);
}

template <typename V>
int GetUVPredictor(const V* const PIK_RESTRICT dc_y, size_t x, size_t y,
                   size_t xsize, intptr_t neg_col_stride,
                   intptr_t neg_row_stride) {
  if (y == 0 || x <= 1 || x + 1 == xsize) {
    return 0;
  }
  const typename Predictors<V>::UV predictor;
  V pred_y[Predictors<V>::kNum];
  V costs[Predictors<V>::kNum];
  predictor(dc_y, neg_col_stride, neg_row_stride, pred_y);
  for (int i = 0; i < Predictors<V>::kNum; ++i) {
    costs[i] = std::abs(dc_y[0] - pred_y[i]);
  }
  return std::min_element(costs, costs + Predictors<V>::kNum) - costs;
}

// Drop-in replacements for dc_predictor functions
template <typename DC>
void SlowShrinkY(const Image<DC>& dc, Image<DC>* const PIK_RESTRICT residuals) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  for (size_t y = 0; y < ysize; y++) {
    const DC* const PIK_RESTRICT row = dc.Row(y);
    DC* const PIK_RESTRICT row_out = residuals->Row(y);
    for (size_t x = 0; x < xsize; x++) {
      DC pred = MinCostPredict(&row[x], x, y, xsize, -1, -dc.PixelsPerRow(),
                               false, 0);
      row_out[x] = row[x] - pred;
    }
  }
}

template <typename DC>
void SlowShrinkUV(const Image<DC>& dc_y, const Image<DC>& dc,
                  Image<DC>* const PIK_RESTRICT residuals) {
  // WARNING: dc and residuals have twice the xsize.
  const size_t xsize = dc_y.xsize();
  const size_t ysize = dc_y.ysize();
  for (size_t y = 0; y < ysize; y++) {
    const DC* const PIK_RESTRICT row = dc.Row(y);
    const DC* const PIK_RESTRICT row_y = dc_y.Row(y);
    DC* const PIK_RESTRICT row_out = residuals->Row(y);
    for (size_t x = 0; x < xsize; x++) {
      int predictor =
          GetUVPredictor(&row_y[x], x, y, xsize, -1, -dc_y.PixelsPerRow());
      for (int i = 0; i < 2; i++) {
        DC pred = MinCostPredict(&row[2 * x + i], x, y, xsize, -2,
                                 -dc.PixelsPerRow(), true, predictor);
        row_out[2 * x + i] = row[2 * x + i] - pred;
      }
    }
  }
}

static inline int Quantize(const int dc, const int quant_step) {
  // TODO(janwas): division via table of magic multipliers+fixups
  const int trunc = dc / quant_step;
  const int mod = dc - (trunc * quant_step);
  // If closer to the end of the interval, round up (+1).
  const int rounded = trunc + (mod > quant_step / 2);
  return rounded;
}

template <typename DC>
void SlowQuantizeY(const Image<DC>& dc, const ImageS& quant_map,
                   Image<DC>* const PIK_RESTRICT residuals) {
  PIK_CHECK(SameSize(*residuals, quant_map));
  PIK_CHECK(SameSize(*residuals, dc));
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  Image<DC> qdc(xsize, ysize);
  for (size_t y = 0; y < ysize; y++) {
    const DC* const PIK_RESTRICT row = dc.Row(y);
    const int16_t* const PIK_RESTRICT row_quant = quant_map.Row(y);
    DC* const PIK_RESTRICT row_out = residuals->Row(y);
    DC* const PIK_RESTRICT row_qdc = qdc.Row(y);
    for (size_t x = 0; x < xsize; x++) {
      const DC pred = MinCostPredict(&row_qdc[x], x, y, xsize, -1,
                                     -qdc.PixelsPerRow(), false, 0);
      const int res = row[x] - pred;
      row_out[x] = Quantize(res, row_quant[x]);
      row_qdc[x] = pred + row_out[x] * row_quant[x];
      // printf("%2zu %2zu: %d (pred %d)=>%d, q %d => %d\n", x, y, row[x], pred,
      //        res, row_quant[x], row_out[x]);
    }
  }
}

template <typename DC>
void SlowDequantizeY(const Image<DC>& residuals, const ImageS& quant_map,
                     Image<DC>* const PIK_RESTRICT dc) {
  PIK_CHECK(SameSize(residuals, quant_map));
  PIK_CHECK(SameSize(residuals, *dc));
  const size_t xsize = dc->xsize();
  const size_t ysize = dc->ysize();
  for (size_t y = 0; y < ysize; y++) {
    DC* const PIK_RESTRICT row = dc->Row(y);
    const DC* const PIK_RESTRICT row_res = residuals.Row(y);
    const DC* const PIK_RESTRICT row_quant = quant_map.Row(y);
    for (size_t x = 0; x < xsize; x++) {
      const DC pred = MinCostPredict(&row[x], x, y, xsize, -1,
                                     -dc->PixelsPerRow(), false, 0);
      const int res = row_res[x] * row_quant[x];
      row[x] = pred + res;
      // printf("%2zu %2zu: %d (pred %d), q %d res %d\n", x, y, row[x], pred,
      //        row_quant[x], res);
    }
  }
}

template <typename Random>
SIMD_ATTR void SlowEqualOnY(const size_t xsize, const size_t ysize,
                            Random* rng) {
  const Rect rect(0, 0, xsize, ysize, xsize, ysize);
  Image<DC> dc(xsize, ysize);
  Image<DC> residuals_fast(xsize, ysize);
  Image<DC> residuals_slow(xsize, ysize);
  GeneratorRandom<DC, Random> generator(rng, 0, 16383);
  GenerateImage(generator, &dc);

  ShrinkY(rect, dc, rect, &residuals_fast);
  SlowShrinkY(dc, &residuals_slow);
  VerifyEqual(residuals_fast, residuals_slow);
}

template <typename Random>
SIMD_ATTR void SlowEqualOnXB(const size_t xsize, const size_t ysize,
                             Random* rng) {
  const Rect rect(0, 0, xsize, ysize, xsize, ysize);
  Image<DC> dc(2 * xsize, ysize);
  Image<DC> dc_y(xsize, ysize);
  Image<DC> residuals_fast(2 * xsize, ysize);
  Image<DC> residuals_slow(2 * xsize, ysize);
  GeneratorRandom<DC, Random> generator(rng, 0, 16383);
  GenerateImage(generator, &dc);
  GenerateImage(generator, &dc_y);

  ShrinkXB(rect, dc_y, dc, &residuals_fast);
  SlowShrinkUV(dc_y, dc, &residuals_slow);
  VerifyEqual(residuals_fast, residuals_slow);
}

SIMD_ATTR void TestSlowAgreeImpl() {
  std::mt19937_64 rng;
  for (size_t ysize = 1; ysize < 8; ++ysize) {
    for (size_t xsize = 1; xsize < 32; ++xsize) {
      SlowEqualOnY(xsize, ysize, &rng);
      SlowEqualOnXB(xsize, ysize, &rng);
    }
  }
}
TEST(DCPredictorTest, TestSlowAgree) { TestSlowAgreeImpl(); }

template <typename Random>
bool VerifySlowQuantizer(const size_t xsize, const size_t ysize, Random* rng) {
  Image<DC> dc(xsize, ysize);
  ImageS quant(xsize, ysize);
  GenerateImage(GeneratorRandom<DC, Random>(rng, 0, 16383), &dc);
  GenerateImage(GeneratorRandom<DC, Random>(rng, 1, 50), &quant);
  Image<DC> res(xsize, ysize);

  SlowQuantizeY(dc, quant, &res);

  Image<DC> recon(xsize, ysize);
  SlowDequantizeY(res, quant, &recon);

  for (size_t y = 0; y < ysize; ++y) {
    const DC* row_dc = dc.Row(y);
    const DC* row_recon = recon.Row(y);
    const int16_t* row_quant = quant.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      const int abs_err = std::abs(row_dc[x] - row_recon[x]);
      if (abs_err >= row_quant[x]) {
        printf("%2zu %2zu: Err %d >= %d\n", x, y, abs_err, row_quant[x]);
        EXPECT_TRUE(false);
        return false;
      }
    }
  }
  return true;
}

TEST(DCPredictorTest, TestSlowQuantizer) {
  std::mt19937_64 rng;
  for (size_t ysize = 1; ysize < 16; ++ysize) {
    for (size_t xsize = 1; xsize < 32; ++xsize) {
      if (!VerifySlowQuantizer(xsize, ysize, &rng)) return;
    }
  }
}

}  // namespace
}  // namespace pik
