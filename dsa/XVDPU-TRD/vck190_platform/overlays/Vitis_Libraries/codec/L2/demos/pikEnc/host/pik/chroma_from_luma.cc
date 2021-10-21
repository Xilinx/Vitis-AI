// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/chroma_from_luma.h"

#include <cstdint>

#include "pik/common.h"
#include "pik/profiler.h"
#include "pik/quantizer.h"
#include "pik/simd/simd.h"

namespace pik {
namespace {

#define PIK_CFL_VERBOSE 0
#if PIK_CFL_VERBOSE
// Which block/coefficient to print (absolute, within entire image)
constexpr size_t kX = 38;
constexpr size_t kY = 8;
constexpr size_t kK = 1;
#endif

// Op{Decorrelate,Restore} are used as template parameters to unify the code
// that applies or removes decorrelation.

struct OpDecorrelate {
  static const char* Name() { return "D"; }

  template <class V>
  SIMD_ATTR V operator()(const V y, const V neg_r, const V x) const {
    // TODO(janwas): FMA
    return x - y * neg_r;
  }
};

struct OpRestore {
  static const char* Name() { return "R"; }

  template <class V>
  SIMD_ATTR V operator()(const V y, const V neg_r, const V x) const {
    // TODO(janwas): FMA
    return y * neg_r + x;
  }
};

struct OpCopy {
  static const char* Name() { return "C"; }

  template <class V>
  SIMD_ATTR V operator()(const V y, const V neg_r, const V x) const {
    return x;
  }
};

// Memory (IIR) for previously restored x/y/b coefficients.
struct Accumulators {
  void Add(const float prev_x, const float prev_y, const float prev_b) {
    sum_xp_ += prev_x * prev_y;
    sum_yp_ += prev_y * prev_y;
    sum_bp_ += prev_b * prev_y;
  }

  // Slightly better (in terms of residual/dct ratio) to avoid slow ramp up
  // via AddNext; in the first call, overwrite immediately.
  bool IsZero() const { return sum_yp_ == 0.0f; }

  void AddNext(const float next_weight, const Accumulators& next) {
    const float prev_weight = 1.0f - next_weight;
    sum_xp_ = next_weight * next.sum_xp_ + prev_weight * sum_xp_;
    sum_yp_ = next_weight * next.sum_yp_ + prev_weight * sum_yp_;
    sum_bp_ = next_weight * next.sum_bp_ + prev_weight * sum_bp_;
  }

  // Computes r_x,r_b = correlation of yp to xp,bp
  void EstimateCorrelation(float* PIK_RESTRICT r0_x,
                           float* PIK_RESTRICT r0_b) const {
    const float rcp0_yp = sum_yp_ == 0.0f ? 0.0f : 1.0f / sum_yp_;
    *r0_x = sum_xp_ * rcp0_yp;
    *r0_b = sum_bp_ * rcp0_yp;
    // No need to clamp - the sum+memory avoids near-zero Y and |r| < 3.
  }

  // private:
  float sum_xp_ = 0.0f;
  float sum_yp_ = 0.0f;
  float sum_bp_ = 0.0f;
};

// Block policy allows the same function to handle AC block or DC coefficient.

// kDCTBlockSize coefficients, first (DC) is invalid.
class BlockAC {
 public:
  static size_t PosFromBX(size_t bx) { return bx * kDCTBlockSize; }

  // Estimates correlation.
  template <class Op>
  static SIMD_ATTR PIK_INLINE void Adaptive(
      const size_t bx, const size_t by,

      const float* PIK_RESTRICT in_y, const float* in_x, const float* in_b,

      const float* PIK_RESTRICT prev_y, const float* PIK_RESTRICT prev_x,
      const float* PIK_RESTRICT prev_b,

      Accumulators* PIK_RESTRICT acc_dc, Accumulators* PIK_RESTRICT acc_ac,
      float* out_x, float* out_b, CFL_Stats* stats) {
    // Can only estimate from low frequencies (189) because the HF predictor
    // changes HF coefficients.
    Accumulators next_ac;
    next_ac.Add(prev_x[1], prev_y[1], prev_b[1]);
    next_ac.Add(prev_x[8], prev_y[8], prev_b[8]);
    next_ac.Add(prev_x[9], prev_y[9], prev_b[9]);

    float r_x, r_b;
    if (acc_ac->IsZero()) {
      *acc_ac = next_ac;
    } else {
      acc_ac->AddNext(0.25f, next_ac);
    }

    acc_ac->EstimateCorrelation(&r_x, &r_b);
    if (stats != nullptr) {
      stats->rx.Notify(r_x);
      stats->rb.Notify(r_b);
    }

    ApplyOp<Op>(bx, by, in_y, in_x, in_b, r_x, r_b, out_x, out_b, stats);
  }

  // Uses predetermined correlation.
  template <class Op>
  static SIMD_ATTR PIK_INLINE void Hardcoded(const size_t bx, const size_t by,
                                             const float* PIK_RESTRICT in_y,
                                             const float* in_x,
                                             const float* in_b, float* out_x,
                                             float* out_b, CFL_Stats* stats) {
    const float r_x = 0.001f;
    const float r_b = 0.93f;
    ApplyOp<Op>(bx, by, in_y, in_x, in_b, r_x, r_b, out_x, out_b, stats);
  }

  static void Quantize(const size_t c, const Quantizer& quantizer,
                       const uint8_t quant_table, const int32_t quant_ac,
                       const float* PIK_RESTRICT from, const size_t from_stride,
                       float* PIK_RESTRICT to, size_t const to_stride) {
    const AcStrategy acs(AcStrategy::Type::DCT, 0);
    PIK_ASSERT(acs.IsFirstBlock());
    quantizer.QuantizeRoundtripBlockAC(
        c, quant_table, quant_ac, acs.GetQuantKind(), acs.covered_blocks_x(),
        acs.covered_blocks_y(), from, from_stride, to, to_stride);
  }

 private:
  // Decorrelates/restores one x and b block. in_x,b may alias out_x,b.
  template <class Op>
  static SIMD_ATTR PIK_INLINE void ApplyOp(const size_t bx, const size_t by,
                                           const float* PIK_RESTRICT in_y,
                                           const float* in_x, const float* in_b,
                                           const float r_x, const float r_b,
                                           float* out_x, float* out_b,
                                           CFL_Stats* stats) {
#if PIK_CFL_VERBOSE
    const float saved_x = in_x[kK];
    const float saved_b = in_b[kK];
#endif

    for (size_t k = 0; k < kDCTBlockSize; ++k) {
      out_x[k] = Op()(in_y[k], r_x, in_x[k]);
      out_b[k] = Op()(in_y[k], r_b, in_b[k]);
    }

#if PIK_CFL_VERBOSE
    if (bx == kX && by == kY) {
      printf("  %s: in %9.4f %9.4f  qY %.4f  r %.4f %.4f  out %.4f %.4f\n",
             Op::Name(), saved_x, saved_b, in_y[kK], r_x, r_b, out_x[kK],
             out_b[kK]);
    }
#endif
  }
};

// Single coefficient
class BlockDC {
 public:
  static size_t PosFromBX(size_t bx) { return bx; }

  // Estimates correlation.
  template <class Op>
  static SIMD_ATTR PIK_INLINE void Adaptive(
      const size_t bx, const size_t by,

      const float* PIK_RESTRICT in_y, const float* in_x, const float* in_b,

      const float* PIK_RESTRICT prev_y, const float* PIK_RESTRICT prev_x,
      const float* PIK_RESTRICT prev_b,

      Accumulators* PIK_RESTRICT acc_dc, Accumulators* PIK_RESTRICT acc_ac,
      float* out_x, float* out_b, CFL_Stats* stats) {
    Accumulators next_dc;
    next_dc.Add(prev_x[0], prev_y[0], prev_b[0]);
    if (acc_dc->IsZero()) {
      *acc_dc = next_dc;
    } else {
      acc_dc->AddNext(0.25f, next_dc);
    }
    float r_x, r_b;
    // Adaptive estimator still helpful for DC (better than constant r).
    acc_dc->EstimateCorrelation(&r_x, &r_b);
    if (stats != nullptr) {
      stats->rx.Notify(r_x);
      stats->rb.Notify(r_b);
    }

    ApplyOp<Op>(bx, by, in_y, in_x, in_b, r_x, r_b, out_x, out_b, stats);
  }

  // Uses predetermined correlation.
  template <class Op>
  static SIMD_ATTR PIK_INLINE void Hardcoded(const size_t bx, const size_t by,
                                             const float* PIK_RESTRICT in_y,
                                             const float* in_x,
                                             const float* in_b, float* out_x,
                                             float* out_b, CFL_Stats* stats) {
    const float r_x = 0.005f;
    const float r_b = 0.93f;
    ApplyOp<Op>(bx, by, in_y, in_x, in_b, r_x, r_b, out_x, out_b, stats);
  }

  static void Quantize(const size_t c, const Quantizer& quantizer,
                       uint8_t quant_table, const int32_t quant_ac,
                       const float* PIK_RESTRICT from, const size_t from_stride,
                       float* PIK_RESTRICT to, size_t const to_stride) {
    // Always use DCT8 quantization kind for DC
    const float mul = quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0] *
                      quantizer.inv_quant_dc();
    *to = quantizer.QuantizeDC(c, *from) * mul;
  }

 private:
  // Decorrelates/restores one x and b block. in_x,b may alias out_x,b.
  template <class Op>
  static SIMD_ATTR PIK_INLINE void ApplyOp(const size_t bx, const size_t by,
                                           const float* PIK_RESTRICT in_y,
                                           const float* in_x, const float* in_b,
                                           const float r_x, const float r_b,
                                           float* out_x, float* out_b,
                                           CFL_Stats* stats) {
    const size_t k = 0;
    out_x[k] = Op()(in_y[k], r_x, in_x[k]);
    out_b[k] = Op()(in_y[k], r_b, in_b[k]);
  }
};

// Fills quantized `residual_xb` and `restored_xb`. `rect_by` < `rect.ysize()`.
template <class Block>
SIMD_ATTR void DecorrelateRow(
    const ImageF& quantized_y, const Image3F& exact_xb, const Rect& rect,
    const size_t rect_by, Accumulators* PIK_RESTRICT acc_dc,
    Accumulators* PIK_RESTRICT acc_ac, Accumulators* PIK_RESTRICT acc_dc2,
    Accumulators* PIK_RESTRICT acc_ac2, const Quantizer& quantizer,
    const uint8_t quant_table, Image3F* PIK_RESTRICT residual_xb,
    Image3F* PIK_RESTRICT restored_xb, CFL_Stats* stats) {
  PROFILER_FUNC;

  // WARNING: do not use rect.*Row because images are in DCT layout.
  const size_t by = rect.y0() + rect_by;
  const float* PIK_RESTRICT row_quantized_y = quantized_y.ConstRow(by);
  const float* PIK_RESTRICT row_exact_x = exact_xb.ConstPlaneRow(0, by);
  const float* PIK_RESTRICT row_exact_b = exact_xb.ConstPlaneRow(2, by);
  const int32_t* PIK_RESTRICT row_quant = quantizer.RawQuantField().Row(by);
  float* PIK_RESTRICT row_residual_x = residual_xb->PlaneRow(0, by);
  float* PIK_RESTRICT row_residual_b = residual_xb->PlaneRow(2, by);
  float* PIK_RESTRICT row_restored_x = restored_xb->PlaneRow(0, by);
  float* PIK_RESTRICT row_restored_b = restored_xb->PlaneRow(2, by);

  // Exact residuals, will be quantized and stored to residual_xb.
  // TODO(janwas): store in separate image to allow repeated quantization
  SIMD_ALIGN float residual_x[kDCTBlockSize];
  SIMD_ALIGN float residual_b[kDCTBlockSize];
  const size_t from_stride = kDCTBlockSize;

  // Leftmost block
  {
    const size_t rect_bx = 0;
    const size_t bx = rect.x0() + rect_bx;
    const size_t pos = Block::PosFromBX(bx);  // coefficient

    if (rect_by == 0) {  // Top-left block: use default correlation
      // Fill residual
      Block::template Hardcoded<OpDecorrelate>(
          bx, by, row_quantized_y + pos, row_exact_x + pos, row_exact_b + pos,
          residual_x, residual_b, stats);

      // residual => row_residual
      Block::Quantize(/*c=*/0, quantizer, quant_table, row_quant[bx],
                      residual_x, from_stride, row_residual_x + pos,
                      residual_xb->PixelsPerRow());
      Block::Quantize(/*c=*/2, quantizer, quant_table, row_quant[bx],
                      residual_b, from_stride, row_residual_b + pos,
                      residual_xb->PixelsPerRow());

      // row_residual => row_restored
      Block::template Hardcoded<OpRestore>(
          bx, by, row_quantized_y + pos, row_residual_x + pos,
          row_residual_b + pos, row_restored_x + pos, row_restored_b + pos,
          stats);
    } else {  // Estimate from north block
      const size_t yp = by - 1;
      const float* PIK_RESTRICT row_prev_y = quantized_y.ConstRow(yp);
      const float* PIK_RESTRICT row_prev_x = restored_xb->ConstPlaneRow(0, yp);
      const float* PIK_RESTRICT row_prev_b = restored_xb->ConstPlaneRow(2, yp);
      // Fill residual
      Block::template Adaptive<OpDecorrelate>(
          bx, by, row_quantized_y + pos, row_exact_x + pos, row_exact_b + pos,
          row_prev_y + pos, row_prev_x + pos, row_prev_b + pos, acc_dc, acc_ac,
          residual_x, residual_b, stats);

      // residual => row_residual
      Block::Quantize(/*c=*/0, quantizer, quant_table, row_quant[bx],
                      residual_x, from_stride, row_residual_x + pos,
                      residual_xb->PixelsPerRow());
      Block::Quantize(/*c=*/2, quantizer, quant_table, row_quant[bx],
                      residual_b, from_stride, row_residual_b + pos,
                      residual_xb->PixelsPerRow());

      // row_residual => row_restored
      Block::template Adaptive<OpRestore>(
          bx, by, row_quantized_y + pos, row_residual_x + pos,
          row_residual_b + pos, row_prev_y + pos, row_prev_x + pos,
          row_prev_b + pos, acc_dc2, acc_ac2, row_restored_x + pos,
          row_restored_b + pos, stats);
    }
  }

  // bx > 0: estimate from west block
  for (size_t rect_bx = 1; rect_bx < rect.xsize(); ++rect_bx) {
    const size_t bx = rect.x0() + rect_bx;
    const size_t pos = Block::PosFromBX(bx);
    const size_t prev = pos - Block::PosFromBX(1);
    // Fill residual
    Block::template Adaptive<OpDecorrelate>(
        bx, by, row_quantized_y + pos, row_exact_x + pos, row_exact_b + pos,
        row_quantized_y + prev, row_restored_x + prev, row_restored_b + prev,
        acc_dc, acc_ac, residual_x, residual_b, stats);

    // residual => row_residual
    Block::Quantize(/*c=*/0, quantizer, quant_table, row_quant[bx], residual_x,
                    from_stride, row_residual_x + pos,
                    residual_xb->PixelsPerRow());
    Block::Quantize(/*c=*/2, quantizer, quant_table, row_quant[bx], residual_b,
                    from_stride, row_residual_b + pos,
                    residual_xb->PixelsPerRow());

    // row_residual => row_restored
    Block::template Adaptive<OpRestore>(
        bx, by, row_quantized_y + pos, row_residual_x + pos,
        row_residual_b + pos, row_quantized_y + prev, row_restored_x + prev,
        row_restored_b + prev, acc_dc2, acc_ac2, row_restored_x + pos,
        row_restored_b + pos, stats);
  }
}

// residual_xb may be aliased with restored_xb.
template <class Block>
SIMD_ATTR void RestoreRow(const ImageF& quantized_y, const Image3F& residual_xb,
                          const Rect& rect, const size_t rect_by,
                          Accumulators* PIK_RESTRICT acc_dc,
                          Accumulators* PIK_RESTRICT acc_ac,
                          Image3F* restored_xb, CFL_Stats* stats) {
  PROFILER_FUNC;

  // WARNING: do not use rect.*Row because images are in DCT layout.
  const size_t by = rect.y0() + rect_by;
  const float* PIK_RESTRICT row_quantized_y = quantized_y.ConstRow(by);
  const float* row_residual_x = residual_xb.PlaneRow(0, by);
  const float* row_residual_b = residual_xb.PlaneRow(2, by);
  float* row_restored_x = restored_xb->PlaneRow(0, by);
  float* row_restored_b = restored_xb->PlaneRow(2, by);

  // Leftmost block
  {
    const size_t rect_bx = 0;
    const size_t bx = rect.x0() + rect_bx;
    const size_t pos = Block::PosFromBX(bx);  // coefficient

    if (rect_by == 0) {  // Top-left block: use default correlation
      Block::template Hardcoded<OpRestore>(
          bx, by, row_quantized_y + pos, row_residual_x + pos,
          row_residual_b + pos, row_restored_x + pos, row_restored_b + pos,
          stats);
    } else {  // Estimate correlation from north block
      const size_t yp = by - 1;
      Block::template Adaptive<OpRestore>(
          bx, by, row_quantized_y + pos, row_residual_x + pos,
          row_residual_b + pos, quantized_y.ConstRow(yp) + pos,
          restored_xb->ConstPlaneRow(0, yp) + pos,
          restored_xb->ConstPlaneRow(2, yp) + pos, acc_dc, acc_ac,
          row_restored_x + pos, row_restored_b + pos, stats);
    }
  }

  // bx > 0: estimate from west block
  for (size_t rect_bx = 1; rect_bx < rect.xsize(); ++rect_bx) {
    const size_t bx = rect.x0() + rect_bx;
    const size_t pos = Block::PosFromBX(bx);  // coefficient
    const size_t prev = pos - Block::PosFromBX(1);
    Block::template Adaptive<OpRestore>(
        bx, by, row_quantized_y + pos, row_residual_x + pos,
        row_residual_b + pos, row_quantized_y + prev, row_restored_x + prev,
        row_restored_b + prev, acc_dc, acc_ac, row_restored_x + pos,
        row_restored_b + pos, stats);
  }
}

// `image` is Image3F/ImageF, either DCT layout or DC.
template <class Block, class ImageT>
void VerifyRectInside(const Rect& rect, const ImageT& image) {
  const size_t x_end = Block::PosFromBX(rect.x0() + rect.xsize());
  const size_t y_end = rect.y0() + rect.ysize();  // blocks
  if (x_end > image.xsize() || y_end > image.ysize()) {
    PIK_ABORT("Rect(blocks) %zu,%zu %zux%zu Image %zux%zu\n", rect.x0(),
              rect.y0(), rect.xsize(), rect.ysize(), image.xsize(),
              image.ysize());
  }
}

template <class Block>
SIMD_ATTR void DecorrelateT(const ImageF& quantized_y, const Image3F& exact_xb,
                            const Rect& rect, const Quantizer& quantizer,
                            size_t quant_table,
                            Image3F* PIK_RESTRICT residual_xb,
                            Image3F* PIK_RESTRICT restored_xb,
                            CFL_Stats* stats) {
  VerifyRectInside<Block>(rect, quantized_y);
  VerifyRectInside<Block>(rect, exact_xb);
  VerifyRectInside<Block>(rect, *residual_xb);
  VerifyRectInside<Block>(rect, *restored_xb);
  // If these are the same, Update() can't compare before and after.
  PIK_ASSERT(&exact_xb != residual_xb);

  Accumulators acc_dc, acc_ac;
  Accumulators acc_dc2, acc_ac2;
  for (size_t rect_by = 0; rect_by < rect.ysize(); ++rect_by) {
    DecorrelateRow<Block>(quantized_y, exact_xb, rect, rect_by, &acc_dc,
                          &acc_ac, &acc_dc2, &acc_ac2, quantizer, quant_table,
                          residual_xb, restored_xb, stats);
  }

  if (stats != nullptr) {
    stats->Update(*residual_xb, *restored_xb, rect, Block::PosFromBX(1));
  }
}

template <class Block>
SIMD_ATTR void RestoreT(const ImageF& quantized_y, const Image3F& residual_xb,
                        const Rect& rect, Image3F* restored_xb,
                        CFL_Stats* stats) {
  VerifyRectInside<Block>(rect, quantized_y);
  VerifyRectInside<Block>(rect, residual_xb);
  VerifyRectInside<Block>(rect, *restored_xb);

  Accumulators acc_dc, acc_ac;
  for (size_t rect_by = 0; rect_by < rect.ysize(); ++rect_by) {
    RestoreRow<Block>(quantized_y, residual_xb, rect, rect_by, &acc_dc, &acc_ac,
                      restored_xb, stats);
  }
}

}  // namespace

SIMD_ATTR void DecorrelateAC(const ImageF& quantized_y, const Image3F& exact_xb,
                             const Rect& rect, const Quantizer& quantizer,
                             uint8_t quant_table,
                             Image3F* PIK_RESTRICT residual_xb,
                             Image3F* PIK_RESTRICT restored_xb,
                             CFL_Stats* stats) {
  DecorrelateT<BlockAC>(quantized_y, exact_xb, rect, quantizer, quant_table,
                        residual_xb, restored_xb, stats);
}

SIMD_ATTR void DecorrelateDC(const ImageF& quantized_y, const Image3F& exact_xb,
                             const Rect& rect, const Quantizer& quantizer,
                             uint8_t quant_table,
                             Image3F* PIK_RESTRICT residual_xb,
                             Image3F* PIK_RESTRICT restored_xb,
                             CFL_Stats* stats) {
  DecorrelateT<BlockDC>(quantized_y, exact_xb, rect, quantizer, quant_table,
                        residual_xb, restored_xb, stats);
}

SIMD_ATTR void RestoreAC(const ImageF& quantized_y, const Image3F& residual_xb,
                         const Rect& rect, Image3F* restored_xb,
                         CFL_Stats* stats) {
  RestoreT<BlockAC>(quantized_y, residual_xb, rect, restored_xb, stats);
}

SIMD_ATTR void RestoreDC(const ImageF& quantized_y, const Image3F& residual_xb,
                         const Rect& rect, Image3F* restored_xb,
                         CFL_Stats* stats) {
  RestoreT<BlockDC>(quantized_y, residual_xb, rect, restored_xb, stats);
}

}  // namespace pik
