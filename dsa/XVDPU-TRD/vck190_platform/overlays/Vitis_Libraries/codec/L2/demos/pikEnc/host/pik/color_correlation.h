// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COLOR_CORRELATION_H_
#define PIK_COLOR_CORRELATION_H_

// Chroma-from-luma, computed using heuristics to determine the best linear
// model for the X and B channels from the Y channel.

#include "pik/bit_reader.h"
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/pik_info.h"
#include "pik/quant_weights.h"

namespace pik {

// Tile is the rectangular grid of blocks that share color correlation
// parameters ("factor_x/b" such that residual_b = blue - Y * factor_b).
constexpr size_t kColorTileDim = 64;

static_assert(kColorTileDim % kBlockDim == 0,
              "Color tile dim should be divisible by block dim");
constexpr size_t kColorTileDimInBlocks = kColorTileDim / kBlockDim;

static_assert(kTileDimInBlocks % kColorTileDimInBlocks == 0,
              "Tile dim should be divisible by color tile dim");

constexpr const int32_t kColorFactorX = 256;
constexpr const int32_t kColorOffsetX = 128;
constexpr const float kColorScaleX = 1.0f / kColorFactorX;

constexpr const int32_t kColorFactorB = 128;
constexpr const int32_t kColorOffsetB = 0;
constexpr const float kColorScaleB = 1.0f / kColorFactorB;

// For dispatching to ColorCorrelationMap::YtoTag overloads.
struct TagX {};
struct TagB {};

struct ColorCorrelationMap {
  ColorCorrelationMap() {}
  ColorCorrelationMap(size_t xsize, size_t ysize)  // pixels
      : ytox_dc(128),
        ytob_dc(120),
        ytox_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)),
        ytob_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)) {
    FillImage(128, &ytox_map);
    FillImage(120, &ytob_map);
  }

  // |y| is scaled by some number calculated from x_factor; consequently,
  // passing 1.0f in place of |y| result will be the scaling factor.
  constexpr static float YtoX(float y, int32_t x_factor) {
    return y * (x_factor - kColorOffsetX) * kColorScaleX;
  }

  // |y| is scaled by some number calculated from b_factor; consequently,
  // passing 1.0f in place of |y| result will be the scaling factor.
  constexpr static float YtoB(float y, int32_t b_factor) {
    return y * (b_factor - kColorOffsetB) * kColorScaleB;
  }

  constexpr static float YtoTag(TagX, float y, int32_t factor) {
    return YtoX(y, factor);
  }
  constexpr static float YtoTag(TagB, float y, int32_t factor) {
    return YtoB(y, factor);
  }

  int32_t ytox_dc;
  int32_t ytob_dc;
  ImageI ytox_map;
  ImageI ytob_map;

  ColorCorrelationMap Copy(const Rect& rect) const {
    ColorCorrelationMap copy;
    copy.ytox_dc = ytox_dc;
    copy.ytob_dc = ytob_dc;
    copy.ytob_map = CopyImage(rect, ytob_map);
    copy.ytox_map = CopyImage(rect, ytox_map);
    return copy;
  }
  ColorCorrelationMap Copy() const { return Copy(Rect(ytox_map)); }
};

SIMD_ATTR void UnapplyColorCorrelationAC(const ColorCorrelationMap& cmap,
                                         const Rect& cmap_rect,
                                         const ImageF& y_plane,
                                         Image3F* coeffs);

template <bool decode>
SIMD_ATTR void ApplyColorCorrelationDC(const ColorCorrelationMap& cmap,
                                       const ImageF& y_plane_dc,
                                       Image3F* coeffs_dc);

void FindBestColorCorrelationMap(const Image3F& opsin,
                                 const DequantMatrices& dequant,
                                 ColorCorrelationMap* cmap);

std::string EncodeColorMap(const ImageI& ac_map, const Rect& rect,
                           const int dc_val, PikImageSizeInfo* info);

bool DecodeColorMap(BitReader* PIK_RESTRICT br, ImageI* PIK_RESTRICT ac_map,
                    int* PIK_RESTRICT dc_val);
}  // namespace pik

#endif  // PIK_COLOR_CORRELATION_H_
