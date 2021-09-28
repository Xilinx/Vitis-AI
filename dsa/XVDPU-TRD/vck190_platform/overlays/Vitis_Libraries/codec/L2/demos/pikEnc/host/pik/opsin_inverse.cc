// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/opsin_inverse.h"

#include <mutex>
#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/compiler_specific.h"
#include "pik/opsin_params.h"
#include "pik/profiler.h"

namespace pik {
namespace {

SIMD_ALIGN float inverse_matrix[9 * SIMD_FULL(float)::N];

SIMD_ATTR void InitInverseMatrix() {
  // Prevent tsan warnings just in case this is called by concurrent decoders.
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  const SIMD_FULL(float) d;
  const float* PIK_RESTRICT inverse = GetOpsinAbsorbanceInverseMatrix();
  for (size_t i = 0; i < 9; ++i) {
    store(set1(d, inverse[i]), d, &inverse_matrix[i * d.N]);
  }
}

// Inverts the pixel-wise RGB->XYB conversion in OpsinDynamicsImage() (including
// the gamma mixing and simple gamma). Avoids clamping to [0, 255] - out of
// (sRGB) gamut values may be in-gamut after transforming to a wider space.
// "inverse_matrix" points to 9 broadcasted vectors, which are the 3x3 entries
// of the (row-major) opsin absorbance matrix inverse. Pre-multiplying its
// entries by c is equivalent to multiplying linear_* by c afterwards.
template <class D, class V>
SIMD_ATTR PIK_INLINE void XybToRgb(D d, const V opsin_x, const V opsin_y,
                                   const V opsin_b,
                                   const float* PIK_RESTRICT inverse_matrix,
                                   V* const PIK_RESTRICT linear_r,
                                   V* const PIK_RESTRICT linear_g,
                                   V* const PIK_RESTRICT linear_b) {
#if SIMD_TARGET_VALUE == SIMD_NONE
  const auto inv_scale_x = set1(d, kInvScaleR);
  const auto inv_scale_y = set1(d, kInvScaleG);
  const auto neg_bias_r = set1(d, kNegOpsinAbsorbanceBiasRGB[0]);
  const auto neg_bias_g = set1(d, kNegOpsinAbsorbanceBiasRGB[1]);
  const auto neg_bias_b = set1(d, kNegOpsinAbsorbanceBiasRGB[2]);
#else
  const auto neg_bias_rgb = load_dup128(d, kNegOpsinAbsorbanceBiasRGB);
  SIMD_ALIGN const float inv_scale_lanes[4] = {kInvScaleR, kInvScaleG};
  const auto inv_scale = load_dup128(d, inv_scale_lanes);
  const auto inv_scale_x = broadcast<0>(inv_scale);
  const auto inv_scale_y = broadcast<1>(inv_scale);
  const auto neg_bias_r = broadcast<0>(neg_bias_rgb);
  const auto neg_bias_g = broadcast<1>(neg_bias_rgb);
  const auto neg_bias_b = broadcast<2>(neg_bias_rgb);
#endif

  // Color space: XYB -> RGB
  const auto gamma_r = inv_scale_x * (opsin_y + opsin_x);
  const auto gamma_g = inv_scale_y * (opsin_y - opsin_x);
  const auto gamma_b = opsin_b;

  // Undo gamma compression: linear = gamma^3 for efficiency.
  const auto gamma_r2 = gamma_r * gamma_r;
  const auto gamma_g2 = gamma_g * gamma_g;
  const auto gamma_b2 = gamma_b * gamma_b;
  const auto mixed_r = mul_add(gamma_r2, gamma_r, neg_bias_r);
  const auto mixed_g = mul_add(gamma_g2, gamma_g, neg_bias_g);
  const auto mixed_b = mul_add(gamma_b2, gamma_b, neg_bias_b);

  // Unmix (multiply by 3x3 inverse_matrix)
  *linear_r = load(d, &inverse_matrix[0 * d.N]) * mixed_r;
  *linear_g = load(d, &inverse_matrix[3 * d.N]) * mixed_r;
  *linear_b = load(d, &inverse_matrix[6 * d.N]) * mixed_r;
  const auto tmp_r = load(d, &inverse_matrix[1 * d.N]) * mixed_g;
  const auto tmp_g = load(d, &inverse_matrix[4 * d.N]) * mixed_g;
  const auto tmp_b = load(d, &inverse_matrix[7 * d.N]) * mixed_g;
  *linear_r = mul_add(load(d, &inverse_matrix[2 * d.N]), mixed_b, *linear_r);
  *linear_g = mul_add(load(d, &inverse_matrix[5 * d.N]), mixed_b, *linear_g);
  *linear_b = mul_add(load(d, &inverse_matrix[8 * d.N]), mixed_b, *linear_b);
  *linear_r += tmp_r;
  *linear_g += tmp_g;
  *linear_b += tmp_b;
}

}  // namespace

SIMD_ATTR void OpsinToLinear(Image3F* PIK_RESTRICT inout) {
  PROFILER_FUNC;
  InitInverseMatrix();
  const size_t xsize = inout->xsize();  // not padded

  for(int task = 0; task < inout->ysize(); ++task) {
        const size_t y = task;

        // Faster than adding via ByteOffset at end of loop.
        float* PIK_RESTRICT row0 = inout->PlaneRow(0, y);
        float* PIK_RESTRICT row1 = inout->PlaneRow(1, y);
        float* PIK_RESTRICT row2 = inout->PlaneRow(2, y);

        const SIMD_FULL(float) d;

        for (size_t x = 0; x < xsize; x += d.N) {
          const auto in_opsin_x = load(d, row0 + x);
          const auto in_opsin_y = load(d, row1 + x);
          const auto in_opsin_b = load(d, row2 + x);
          PIK_COMPILER_FENCE;
          SIMD_FULL(float)::V linear_r, linear_g, linear_b;
          XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, inverse_matrix,
                   &linear_r, &linear_g, &linear_b);

          store(linear_r, d, row0 + x);
          store(linear_g, d, row1 + x);
          store(linear_b, d, row2 + x);
        }
  }
}

SIMD_ATTR void OpsinToLinear(const Image3F& opsin, const Rect& rect_out,
                             Image3F* PIK_RESTRICT linear) {
  PROFILER_ZONE("OpsinToLinear(Rect)");
  InitInverseMatrix();
  PIK_ASSERT(linear->xsize() != 0);
  // Opsin is padded to blocks; only produce valid output pixels.
  const size_t xsize = rect_out.xsize();
  const size_t ysize = rect_out.ysize();
  PIK_ASSERT(xsize <= opsin.xsize());
  PIK_ASSERT(ysize <= opsin.ysize());

  for (size_t y = 0; y < ysize; ++y) {
    // Faster than adding via ByteOffset at end of loop.
    const float* PIK_RESTRICT row_opsin_x = opsin.ConstPlaneRow(0, y);
    const float* PIK_RESTRICT row_opsin_y = opsin.ConstPlaneRow(1, y);
    const float* PIK_RESTRICT row_opsin_b = opsin.ConstPlaneRow(2, y);

    float* PIK_RESTRICT row_linear_r = rect_out.PlaneRow(linear, 0, y);
    float* PIK_RESTRICT row_linear_g = rect_out.PlaneRow(linear, 1, y);
    float* PIK_RESTRICT row_linear_b = rect_out.PlaneRow(linear, 2, y);

    const SIMD_FULL(float) d;

    for (size_t x = 0; x < xsize; x += d.N) {
      const auto in_opsin_x = load(d, row_opsin_x + x);
      const auto in_opsin_y = load(d, row_opsin_y + x);
      const auto in_opsin_b = load(d, row_opsin_b + x);
      PIK_COMPILER_FENCE;
      SIMD_FULL(float)::V linear_r, linear_g, linear_b;
      XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, inverse_matrix, &linear_r,
               &linear_g, &linear_b);

      store(linear_r, d, row_linear_r + x);
      store(linear_g, d, row_linear_g + x);
      store(linear_b, d, row_linear_b + x);
    }
  }
}

}  // namespace pik
