// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/adaptive_quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <sys/time.h>
#include <iostream>
#include "pik/quant_weights.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/ac_strategy.h"
#include "pik/approx_cube_root.h"
#include "pik/butteraugli_comparator.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/compressed_dc.h"
#include "pik/compressed_image.h"
#include "pik/dct.h"
#include "pik/entropy_coder.h"
#include "pik/gauss_blur.h"
#include "pik/gradient_map.h"
#include "pik/image.h"
#include "pik/opsin_inverse.h"
#include "pik/profiler.h"
#include "pik/status.h"

bool FLAGS_log_search_state = false;
// If true, prints the quantization maps at each iteration.
bool FLAGS_dump_quant_state = false;

namespace pik {
namespace {

static const double kQuant64[64] = {
    0.0,
    1.3209836994457049,
    1.3209836994457049,
    3.8093709838312875,
    3.8093709838312875,
    3.24478652961692,
    3.2932050795619587,
    3.2773870962477889,
    3.2833062377265221,
    0.86300641731251715,
    0.81525767385332304,
    2.509455259261768,
    2.5512715939619355,
    1.1009460471286041,
    1.0472242371200757,
    1.2132776894507991,
    1.1836672984883603,
    2.0934299406343144,
    2.1313160506404598,
    0.89190317505002714,
    0.840986927484077,
    0.70197816830313797,
    0.76186836723000584,
    1.3680171732844342,
    1.3437082798869822,
    1.2470683713759048,
    1.1765069429040906,
    0.35169288065810422,
    0.41178711526516226,
    0.53498575397594228,
    0.60137373453997023,
    0.70810156203993546,
    0.68693731722531248,
    0.27052366537985795,
    0.22640452462879396,
    0.198291255325797,
    0.22234508352308596,
    0.30485790798913998,
    0.28530044891325101,
    0.29524052213046498,
    0.33185918780276397,
    0.353582036654603,
    0.305955548639682,
    0.365259530060446,
    0.333159510814334,
    0.363133568767434,
    0.334161790012618,
    0.389194124900511,
    0.349326306148990,
    0.390310895605386,
    0.408666924454222,
    0.335930464190049,
    0.359313000261458,
    0.381109877480420,
    0.392933763109596,
    0.359529015172913,
    0.347676628893596,
    0.370974565818013,
    0.350361463992334,
    0.338064798002449,
    0.336743523710490,
    0.296631529585931,
    0.304517245589665,
    0.302956514467806,
};

// Increase precision in 8x8 blocks that are complicated in DCT space.
SIMD_ATTR void DctModulation(const ImageF& xyb, ImageF* out) {
  PIK_ASSERT((xyb.xsize() + 7) / 8 == out->xsize());
  PIK_ASSERT((xyb.ysize() + 7) / 8 == out->ysize());
  const int32_t* natural_coeff_order = NaturalCoeffOrder();
  float dct_rescale[64] = {0};
  {
    const float* dct_scale = DCTScales<8>();
    for (int i = 0; i < 64; ++i) {
      dct_rescale[i] = dct_scale[i / 8] * dct_scale[i % 8];
    }
  }
  for (int y = 0; y < xyb.ysize(); y += 8) {
    float* const PIK_RESTRICT row_out = out->Row(y / 8);
    for (int x = 0; x < xyb.xsize(); x += 8) {
      SIMD_ALIGN float dct[64] = {0};
      for (int dy = 0; dy < 8; ++dy) {
        int yclamp = std::min<int>(y + dy, xyb.ysize() - 1);
        const float* const PIK_RESTRICT row_in = xyb.Row(yclamp);
        for (int dx = 0; dx < 8; ++dx) {
          int xclamp = std::min<int>(x + dx, xyb.xsize() - 1);
          dct[dy * 8 + dx] = row_in[xclamp];
        }
      }
      ComputeTransposedScaledDCT<8>()(FromBlock<8>(dct), ToBlock<8>(dct));
      double entropyQL2 = 0;
      double entropyQL4 = 0;
      double entropyQL8 = 0;
      for (int k = 1; k < 64; ++k) {
        int i = natural_coeff_order[k];
        const float scale = dct_rescale[i];
        double v = dct[i] * scale;
        v *= v;
        static const double kPow = 1.923527252414339;
        double q = pow(kQuant64[k], kPow);
        entropyQL2 += q * v;
        v *= v;
        entropyQL4 += q * v;
        v *= v;
        entropyQL8 += q * v;
      }
      entropyQL2 = std::sqrt(entropyQL2);
      entropyQL4 = std::sqrt(std::sqrt(entropyQL4));
      entropyQL8 = std::pow(entropyQL8, 0.125);
      static const double mulQL2 = -0.00072185944355851461;
      static const double mulQL4 = -1.1783135317666862;
      static const double mulQL8 = 0.29099162398822259;
      double v =
          mulQL2 * entropyQL2 + mulQL4 * entropyQL4 + mulQL8 * entropyQL8;
      double kMul = 1.1555549005271522;
      row_out[x / 8] += kMul * v;
    }
  }
}

// Increase precision in 8x8 blocks that have high dynamic range.
void RangeModulation(const ImageF& xyb, ImageF* out) {
  PIK_ASSERT((xyb.xsize() + 7) / 8 == out->xsize());
  PIK_ASSERT((xyb.ysize() + 7) / 8 == out->ysize());
  for (int y = 0; y < xyb.ysize(); y += 8) {
    float* const PIK_RESTRICT row_out = out->Row(y / 8);
    for (int x = 0; x < xyb.xsize(); x += 8) {
      float minval = 1e30;
      float maxval = -1e30;
      for (int dy = 0; dy < 8 && y + dy < xyb.ysize(); ++dy) {
        const float* const PIK_RESTRICT row_in = xyb.Row(y + dy);
        for (int dx = 0; dx < 8 && x + dx < xyb.xsize(); ++dx) {
          float v = row_in[x + dx];
          if (minval > v) {
            minval = v;
          }
          if (maxval < v) {
            maxval = v;
          }
        }
      }
      float range = maxval - minval;
      static const double mul = 0.67975181715504351;
      row_out[x / 8] += mul * range;
    }
  }
}

// Change precision in 8x8 blocks that have high frequency content.
void HfModulation(const ImageF& xyb, ImageF* out) {
  PIK_ASSERT((xyb.xsize() + 7) / 8 == out->xsize());
  PIK_ASSERT((xyb.ysize() + 7) / 8 == out->ysize());
  for (int y = 0; y < xyb.ysize(); y += 8) {
    float* const PIK_RESTRICT row_out = out->Row(y / 8);
    for (int x = 0; x < xyb.xsize(); x += 8) {
      float sum = 0;
      int n = 0;
      for (int dy = 0; dy < 8 && y + dy < xyb.ysize(); ++dy) {
        const float* const PIK_RESTRICT row_in = xyb.Row(y + dy);
        for (int dx = 0; dx < 7 && x + dx + 1 < xyb.xsize(); ++dx) {
          float v = fabs(row_in[x + dx] - row_in[x + dx + 1]);
          sum += v;
          ++n;
        }
      }
      for (int dy = 0; dy < 7 && y + dy + 1 < xyb.ysize(); ++dy) {
        const float* const PIK_RESTRICT row_in = xyb.Row(y + dy);
        const float* const PIK_RESTRICT row_in_next = xyb.Row(y + dy + 1);
        for (int dx = 0; dx < 8 && x + dx < xyb.xsize(); ++dx) {
          float v = fabs(row_in[x + dx] - row_in_next[x + dx]);
          sum += v;
          ++n;
        }
      }
      if (n != 0) {
        sum /= n;
      }
      static const double kMul = 0.70743567045382239;
      sum *= kMul;
      row_out[x / 8] += sum;
    }
  }
}

// We want multiplicative quantization field, so everything until this
// point has been modulating the exponent.
void Exp(ImageF* out) {
  for (int y = 0; y < out->ysize(); ++y) {
    float* const PIK_RESTRICT row_out = out->Row(y);
    for (int x = 0; x < out->xsize(); ++x) {
      row_out[x] = exp(row_out[x]);
    }
  }
}

static double SimpleGamma(double v) {
  // A simple HDR compatible gamma function.
  // mul and mul2 represent a scaling difference between pik and butteraugli.
  static const double mul = 103.34350600371506;
  static const double mul2 = 1.0 / (67.797075768826289);

  v *= mul;

  static const double kRetMul = mul2 * 18.6580932135;
  static const double kRetAdd = mul2 * -20.2789020414;
  static const double kVOffset = 7.14672470003;

  if (v < 0) {
    // This should happen rarely, but may lead to a NaN, which is rather
    // undesirable. Since negative photons don't exist we solve the NaNs by
    // clamping here.
    v = 0;
  }
  return kRetMul * log(v + kVOffset) + kRetAdd;
}

static double RatioOfCubicRootToSimpleGamma(double v) {
  // The opsin space in pik is the cubic root of photons, i.e., v * v * v
  // is related to the number of photons.
  //
  // SimpleGamma(v * v * v) is the psychovisual space in butteraugli.
  // This ratio allows quantization to move from pik's opsin space to
  // butteraugli's log-gamma space.
  return v / SimpleGamma(v * v * v);
}

ImageF DiffPrecompute(const Image3F& xyb, float cutoff) {
  PROFILER_ZONE("aq DiffPrecompute");
  PIK_ASSERT(xyb.xsize() > 1);
  PIK_ASSERT(xyb.ysize() > 1);
  ImageF result(xyb.xsize(), xyb.ysize());
  static const double mul0 = 0.046650519741099357;

  // PIK's gamma is 3.0 to be able to decode faster with two muls.
  // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
  // We approximate the gamma difference by adding one cubic root into
  // the adaptive quantization. This gives us a total gamma of 2.6666
  // for quantization uses.
  static const double match_gamma_offset = 0.55030107636310233;
  static const float kOverWeightBorders = 1.4;
  size_t x1, y1;
  size_t x2, y2;
  for (size_t y = 0; y + 1 < xyb.ysize(); ++y) {
    if (y + 1 < xyb.ysize()) {
      y2 = y + 1;
    } else if (y > 0) {
      y2 = y - 1;
    } else {
      y2 = y;
    }
    if (y == 0 && xyb.ysize() >= 2) {
      y1 = y + 1;
    } else if (y > 0) {
      y1 = y - 1;
    } else {
      y1 = y;
    }
    const float* PIK_RESTRICT row_in = xyb.PlaneRow(1, y);
    const float* PIK_RESTRICT row_in1 = xyb.PlaneRow(1, y1);
    const float* PIK_RESTRICT row_in2 = xyb.PlaneRow(1, y2);
    float* const PIK_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x + 1 < xyb.xsize(); ++x) {
      if (x + 1 < xyb.xsize()) {
        x2 = x + 1;
      } else if (x > 0) {
        x2 = x - 1;
      } else {
        x2 = x;
      }
      if (x == 0 && xyb.xsize() >= 2) {
        x1 = x + 1;
      } else if (x > 0) {
        x1 = x - 1;
      } else {
        x1 = x;
      }
      float diff =
          mul0 *
          (fabs(row_in[x] - row_in[x2]) + fabs(row_in[x] - row_in2[x]) +
           fabs(row_in[x] - row_in[x1]) + fabs(row_in[x] - row_in1[x]) +
           3 * (fabs(row_in2[x] - row_in1[x]) + fabs(row_in[x1] - row_in[x2])));
      diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
      row_out[x] = std::min(cutoff, diff);
    }
    // Last pixel of the row.
    {
      const size_t x = xyb.xsize() - 1;
      float diff =
          kOverWeightBorders * 2.0 * mul0 * (fabs(row_in[x] - row_in2[x]));
      diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
      row_out[x] = std::min(cutoff, diff);
    }
  }
  // Last row.
  {
    const size_t y = xyb.ysize() - 1;
    const float* const PIK_RESTRICT row_in = xyb.PlaneRow(1, y);
    float* const PIK_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x + 1 < xyb.xsize(); ++x) {
      const size_t x2 = x + 1;
      float diff =
          kOverWeightBorders * 2.0 * mul0 * fabs(row_in[x] - row_in[x2]);
      diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
      row_out[x] = std::min(cutoff, diff);
    }
    // Last pixel of the last row.
    {
      const size_t x = xyb.xsize() - 1;
      row_out[x] = row_out[x - 1];
    }
  }
  return result;
}

// Expand the average of last three pixels to form a larger image.
ImageF Expand(const ImageF& img, size_t out_xsize, size_t out_ysize) {
  PIK_ASSERT(img.xsize() > 0);
  PIK_ASSERT(img.ysize() > 0);
  ImageF out(out_xsize, out_ysize);
  // Expand to columns on right.
  for (size_t y = 0; y < img.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = img.Row(y);
    float* const PIK_RESTRICT row_out = out.Row(y);
    memcpy(row_out, row_in, img.xsize() * sizeof(row_out[0]));
    float lastval = row_in[img.xsize() - 1];
    if (img.xsize() >= 3) {
      lastval += row_in[img.xsize() - 3];
      lastval += row_in[img.xsize() - 2];
      lastval *= (1.0 / 3);
    } else if (img.xsize() >= 2) {
      lastval += row_in[img.xsize() - 2];
      lastval *= 0.5;
    }
    for (size_t x = img.xsize(); x < out_xsize; ++x) {
      row_out[x] = lastval;
    }
  }
  // Expand to rows at bottom.
  if (img.ysize() != out_ysize) {
    for (size_t x = 0; x < out_xsize; ++x) {
      const size_t ys = img.ysize();
      float lastval = out.Row(ys - 1)[x];
      if (ys >= 3) {
        lastval += out.Row(ys - 2)[x];
        lastval += out.Row(ys - 3)[x];
        lastval *= (1.0 / 3);
      } else if (ys >= 2) {
        lastval += out.Row(ys - 2)[x];
        lastval *= 0.5;
      }
      for (size_t y = img.ysize(); y < out_ysize; ++y) {
        out.Row(y)[x] = lastval;
      }
    }
  }
  return out;
}

ImageF ComputeMask(const ImageF& diffs) {
  static const float kBase = 1.329262607500535;
  static const float kMul1 = 0.010994306366172898;
  static const float kOffset1 = 0.00683227084849159;
  static const float kMul2 = -0.1949226495025296;
  static const float kOffset2 = 0.075052668223305155;
  ImageF out(diffs.xsize(), diffs.ysize());
  for (int y = 0; y < diffs.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = diffs.Row(y);
    float* const PIK_RESTRICT row_out = out.Row(y);
    for (int x = 0; x < diffs.xsize(); ++x) {
      const float val = row_in[x];
      // Avoid division by zero.
      double div = std::max<double>(val + kOffset1, 1e-3);
      row_out[x] = kBase + kMul1 / div + kMul2 / (val * val + kOffset2);
    }
  }
  return out;
}

ImageF TileDistMap(const ImageF& distmap, int tile_size, int margin,
                   const AcStrategyImage& ac_strategy) {
  PROFILER_FUNC;
  const int tile_xsize = (distmap.xsize() + tile_size - 1) / tile_size;
  const int tile_ysize = (distmap.ysize() + tile_size - 1) / tile_size;
  ImageF tile_distmap(tile_xsize, tile_ysize);
  size_t distmap_stride = tile_distmap.PixelsPerRow();
  for (int tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(tile_y);
    float* PIK_RESTRICT dist_row = tile_distmap.Row(tile_y);
    for (int tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      AcStrategy acs = ac_strategy_row[tile_x];
      if (!acs.IsFirstBlock()) continue;
      int this_tile_xsize = acs.covered_blocks_x() * tile_size;
      int this_tile_ysize = acs.covered_blocks_y() * tile_size;
      int y_begin = std::max<int>(0, tile_size * tile_y - margin);
      int y_end = std::min<int>(distmap.ysize(),
                                tile_size * tile_y + this_tile_ysize + margin);
      int x_begin = std::max<int>(0, tile_size * tile_x - margin);
      int x_end = std::min<int>(distmap.xsize(),
                                tile_size * tile_x + this_tile_xsize + margin);
      float dist_norm = 0.0;
      double pixels = 0;
      for (int y = y_begin; y < y_end; ++y) {
        float ymul = 1.0;
        static const float kBorderMul = 0.98f;
        static const float kCornerMul = 0.7f;
        if (margin != 0 && (y == y_begin || y == y_end - 1)) {
          ymul = kBorderMul;
        }
        const float* const PIK_RESTRICT row = distmap.Row(y);
        for (int x = x_begin; x < x_end; ++x) {
          float xmul = ymul;
          if (margin != 0 && (x == x_begin || x == x_end - 1)) {
            if (xmul == 1.0) {
              xmul = kBorderMul;
            } else {
              xmul = kCornerMul;
            }
          }
          float v = row[x];
          v *= v;
          v *= v;
          v *= v;
          v *= v;
          dist_norm += xmul * v;
          pixels += xmul;
        }
      }
      if (pixels == 0) pixels = 1;
      // 16th norm is less than the max norm, we reduce the difference
      // with this normalization factor.
      static const double kTileNorm = 1.2;
      const double tile_dist = kTileNorm * pow(dist_norm / pixels, 1.0 / 16);
      dist_row[tile_x] = tile_dist;
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          dist_row[tile_x + distmap_stride * iy + ix] = tile_dist;
        }
      }
    }
  }
  return tile_distmap;
}

ImageF DistToPeakMap(const ImageF& field, float peak_min, int local_radius,
                     float peak_weight) {
  ImageF result(field.xsize(), field.ysize());
  FillImage(-1.0f, &result);
  for (int y0 = 0; y0 < field.ysize(); ++y0) {
    for (int x0 = 0; x0 < field.xsize(); ++x0) {
      int x_min = std::max(0, x0 - local_radius);
      int y_min = std::max(0, y0 - local_radius);
      int x_max = std::min<int>(field.xsize(), x0 + 1 + local_radius);
      int y_max = std::min<int>(field.ysize(), y0 + 1 + local_radius);
      float local_max = peak_min;
      for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
          local_max = std::max(local_max, field.Row(y)[x]);
        }
      }
      if (field.Row(y0)[x0] >
          (1.0f - peak_weight) * peak_min + peak_weight * local_max) {
        for (int y = y_min; y < y_max; ++y) {
          for (int x = x_min; x < x_max; ++x) {
            float dist = std::max(std::abs(y - y0), std::abs(x - x0));
            float cur_dist = result.Row(y)[x];
            if (cur_dist < 0.0 || cur_dist > dist) {
              result.Row(y)[x] = dist;
            }
          }
        }
      }
    }
  }
  return result;
}

bool AdjustQuantVal(float* const PIK_RESTRICT q, const float d,
                    const float factor, const float quant_max) {
  if (*q >= 0.999f * quant_max) return false;
  const float inv_q = 1.0f / *q;
  const float adj_inv_q = inv_q - factor / (d + 1.0f);
  *q = 1.0f / std::max(1.0f / quant_max, adj_inv_q);
  return true;
}

void DumpHeatmap(const PikInfo* info, const std::string& label,
                 const ImageF& image, float good_threshold,
                 float bad_threshold) {
  Image3B heatmap =
      butteraugli::CreateHeatMapImage(image, good_threshold, bad_threshold);
  char filename[200];
  snprintf(filename, sizeof(filename), "%s%05d", label.c_str(),
           info->num_butteraugli_iters);
  info->DumpImage(filename, heatmap);
}

void DumpHeatmaps(const PikInfo* info, float ba_target,
                  const ImageF& quant_field, const ImageF& tile_heatmap) {
  if (!WantDebugOutput(info)) return;
  ImageF inv_qmap(quant_field.xsize(), quant_field.ysize());
  for (size_t y = 0; y < quant_field.ysize(); ++y) {
    const float* PIK_RESTRICT row_q = quant_field.ConstRow(y);
    float* PIK_RESTRICT row_inv_q = inv_qmap.Row(y);
    for (size_t x = 0; x < quant_field.xsize(); ++x) {
      row_inv_q[x] = 1.0f / row_q[x];  // never zero
    }
  }
  DumpHeatmap(info, "quant_heatmap", inv_qmap, 4.0f * ba_target,
              6.0f * ba_target);
  DumpHeatmap(info, "tile_heatmap", tile_heatmap, ba_target, 1.5f * ba_target);
}

void AdjustQuantField(const AcStrategyImage& ac_strategy, ImageF* quant_field) {
  // Replace the whole quant_field in non-8x8 blocks with the maximum of each
  // 8x8 block.
  size_t stride = quant_field->PixelsPerRow();
  for (size_t y = 0; y < quant_field->ysize(); ++y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(y);
    float* PIK_RESTRICT quant_row = quant_field->Row(y);
    for (size_t x = 0; x < quant_field->xsize(); ++x) {
      AcStrategy acs = ac_strategy_row[x];
      PIK_ASSERT(x + acs.covered_blocks_x() <= quant_field->xsize());
      PIK_ASSERT(y + acs.covered_blocks_y() <= quant_field->ysize());
      float max = quant_row[x];
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          max = std::max(quant_row[x + ix + iy * stride], max);
        }
      }
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          quant_row[x + ix + iy * stride] = max;
        }
      }
    }
  }
}

Image3F RoundtripImage(
    const CompressParams& cparams, const FrameHeader& frame_header,
    const GroupHeader& header, const Image3F& opsin_orig, const Image3F& opsin,
    const ColorCorrelationMap& full_cmap,
    const BlockDictionary& block_dictionary, const AcStrategyImage& ac_strategy,
    const ImageB& sigma_lut_ids, const Quantizer& quantizer,
    const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    MultipassManager* multipass_manager) {

  PROFILER_ZONE("enc roundtrip");
  FrameDecCache frame_dec_cache;
  frame_dec_cache.ac_strategy = ac_strategy.Copy();
  PIK_ASSERT(opsin.ysize() % kBlockDim == 0);
  frame_dec_cache.raw_quant_field = CopyImage(quantizer.RawQuantField());
  frame_dec_cache.ar_sigma_lut_ids = CopyImage(sigma_lut_ids);

  const size_t xsize_groups = DivCeil(opsin.xsize(), kGroupDim);
  const size_t ysize_groups = DivCeil(opsin.ysize(), kGroupDim);
  const size_t num_groups = xsize_groups * ysize_groups;

  PikInfo aux_out;

  FrameEncCache frame_enc_cache;
  frame_enc_cache.dequant_control_field = CopyImage(dequant_control_field);
  memcpy(frame_enc_cache.dequant_map, dequant_map,
         sizeof(uint8_t) * 256 * kMaxQuantControlFieldValue);
  frame_dec_cache.dequant_control_field = CopyImage(dequant_control_field);
  memcpy(frame_dec_cache.dequant_map, dequant_map,
         sizeof(uint8_t) * 256 * kMaxQuantControlFieldValue);
  InitializeFrameEncCache(frame_header, opsin, ac_strategy, quantizer,
                          full_cmap, block_dictionary, &frame_enc_cache,
                          &aux_out);

  frame_dec_cache.dc = CopyImage(frame_enc_cache.dc_dec);
  frame_dec_cache.gradient = std::move(frame_enc_cache.gradient);

  std::vector<MultipassHandler*> handlers(num_groups);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    const size_t gx = group_index % xsize_groups;
    const size_t gy = group_index / xsize_groups;
    const Rect rect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim,
                    opsin.xsize(), opsin.ysize());
    handlers[group_index] =
        multipass_manager->GetGroupHandler(group_index, rect);
  }

  Image3F idct(opsin.xsize(), opsin.ysize());

  std::vector<GroupDecCache> group_dec_caches(1);

  std::vector<PikInfo> aux_outs(1);

  for(int group_index = 0; group_index < num_groups; ++group_index) {
    GroupDecCache* PIK_RESTRICT group_dec_cache = &group_dec_caches[0];
    PikInfo* my_aux_out = &aux_outs[0];
    MultipassHandler* handler = handlers[group_index];
    const Rect& group_rect = handler->PaddedGroupRect();
    Rect block_group_rect = handler->BlockGroupRect();
    EncCache cache;
    InitializeEncCache(frame_header, header, frame_enc_cache, group_rect,
                       &cache);
    Quantizer quant = quantizer.Copy(block_group_rect);

    Rect group_in_color_tiles(
        block_group_rect.x0() / kColorTileDimInBlocks,
        block_group_rect.y0() / kColorTileDimInBlocks,
        DivCeil(block_group_rect.xsize(), kColorTileDimInBlocks),
        DivCeil(block_group_rect.ysize(), kColorTileDimInBlocks));

    ComputeCoefficients(quant, full_cmap, group_in_color_tiles,
                        frame_enc_cache, &cache, my_aux_out);

    InitializeDecCache(frame_dec_cache, group_rect, group_dec_cache);
    DequantImageAC(quant, full_cmap, group_in_color_tiles, cache.ac,
                   &frame_dec_cache, group_dec_cache, group_rect, my_aux_out);
    ReconOpsinImage(frame_header, header, quant, block_group_rect,
                    &frame_dec_cache, group_dec_cache, &idct, group_rect,
                    my_aux_out);
  }

  aux_out.Assimilate(aux_outs[0]);

  multipass_manager->RestoreOpsin(&idct);
  // Fine to do a PIK_ASSERT instead of error handling, since this only happens
  // on the encoder side where we can't be fed with invalid data.
  PIK_CHECK(FinalizeFrameDecoding(&idct, opsin_orig.xsize(), opsin_orig.ysize(),
                                  frame_header, NoiseParams(), quantizer,
                                  block_dictionary, &frame_dec_cache));
  return idct;
}

static const float kDcQuantPow = 0.57840232344431763;
static const float kDcQuant = 0.74852919562896747;
static const float kAcQuant = 0.97136686727219523;

void FindBestQuantization(
    const Image3F& opsin_orig, const Image3F& opsin_arg,
    const CompressParams& cparams, const FrameHeader& frame_header,
    const GroupHeader& header, float butteraugli_target,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
    ImageF& quant_field, Quantizer* quantizer,
    const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    PikInfo* aux_out, MultipassManager* multipass_manager, double rescale) {
  const float intensity_multiplier = cparams.GetIntensityMultiplier();
  ButteraugliComparator comparator(opsin_orig, cparams.hf_asymmetry,
                                   intensity_multiplier);
  const float initial_quant_dc =
      InitialQuantDC(butteraugli_target, intensity_multiplier);

  AdjustQuantField(ac_strategy, &quant_field);

  ImageF tile_distmap;
  ImageF tile_distmap_localopt;
  ImageF initial_quant_field = CopyImage(quant_field);
  ImageF last_quant_field = CopyImage(initial_quant_field);
  ImageF last_tile_distmap_localopt;

  float initial_qf_min, initial_qf_max;
  ImageMinMax(initial_quant_field, &initial_qf_min, &initial_qf_max);

  float initial_qf_ratio = initial_qf_max / initial_qf_min;
  float qf_max_deviation_low = std::sqrt(250 / initial_qf_ratio);
  float asymmetry = 2;
  if (qf_max_deviation_low < asymmetry) asymmetry = qf_max_deviation_low;
  float qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
  float qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

  PIK_ASSERT(qf_higher / qf_lower < 253);

  constexpr int kOriginalComparisonRound = 5;
  constexpr float kMaximumDistanceIncreaseFactor = 1.015;

  for (int i = 0; i < cparams.max_butteraugli_iters + 1; ++i) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }

    if (quantizer->SetQuantField(initial_quant_dc, QuantField(quant_field))) {

      Image3F linear = RoundtripImage(
          cparams, frame_header, header, opsin_orig, opsin_arg, cmap,
          block_dictionary, ac_strategy, ar_sigma_lut_ids, *quantizer,
          dequant_control_field, dequant_map, multipass_manager);

      PROFILER_ZONE("enc Butteraugli");

      comparator.Compare(linear);

      static const int kMargins[100] = {0, 0, 0, 1, 2, 1, 1, 1, 0};
      tile_distmap =
          TileDistMap(comparator.distmap(), 8, kMargins[i], ac_strategy);
      tile_distmap_localopt =
          TileDistMap(comparator.distmap(), 8, 2, ac_strategy);
      if (WantDebugOutput(aux_out)) {
        DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
        ++aux_out->num_butteraugli_iters;
      }

      if (FLAGS_log_search_state) {
        float minval, maxval;
        ImageMinMax(quant_field, &minval, &maxval);
        printf("\nButteraugli iter: %d/%d\n", i, cparams.max_butteraugli_iters);
        printf("Butteraugli distance: %f\n", comparator.distance());
        printf("quant range: %f ... %f  DC quant: %f\n", minval, maxval,
               initial_quant_dc);
        if (FLAGS_dump_quant_state) {
          quantizer->DumpQuantizationMap();
        }
      }
    }

    if (i > kOriginalComparisonRound) {
      // Undo last round if it made things worse (i.e. increased the quant value
      // AND the distance in nearby pixels by at least some percentage).
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        float* const PIK_RESTRICT row_q = quant_field.Row(y);
        const float* const PIK_RESTRICT row_dist = tile_distmap_localopt.Row(y);
        const float* const PIK_RESTRICT row_last_dist =
            last_tile_distmap_localopt.Row(y);
        const float* const PIK_RESTRICT row_last_q = last_quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          if (row_q[x] > row_last_q[x] &&
              row_dist[x] > kMaximumDistanceIncreaseFactor * row_last_dist[x]) {
            row_q[x] = row_last_q[x];
          }
        }
      }
    }
    last_quant_field = CopyImage(quant_field);
    last_tile_distmap_localopt = CopyImage(tile_distmap_localopt);
    if (i == cparams.max_butteraugli_iters) break;

    double kPow[8] = {
        0.97524596113492301,
        1.0424361904568509,
        0.64984804448911193,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    double kPowMod[8] = {
        0.011236980155043978,
        0.0061256294105472651,
        -0.0030115055086858242,
        0.06929488142351059,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    if (i == kOriginalComparisonRound) {
      // Don't allow optimization to make the quant field a lot worse than
      // what the initial guess was. This allows the AC field to have enough
      // precision to reduce the oscillations due to the dc reconstruction.
      double kInitMul = 0.6;
      const double kOneMinusInitMul = 1.0 - kInitMul;
      for (int y = 0; y < quant_field.ysize(); ++y) {
        float* const PIK_RESTRICT row_q = quant_field.Row(y);
        const float* const PIK_RESTRICT row_init = initial_quant_field.Row(y);
        for (int x = 0; x < quant_field.xsize(); ++x) {
          double clamp = kOneMinusInitMul * row_q[x] + kInitMul * row_init[x];
          if (row_q[x] < clamp) {
            row_q[x] = clamp;
            if (row_q[x] > qf_higher) row_q[x] = qf_higher;
            if (row_q[x] < qf_lower) row_q[x] = qf_lower;
          }
        }
      }
    }

    double cur_pow = 0.0;
    if (i < 7) {
      cur_pow = kPow[i] + (butteraugli_target - 1.0) * kPowMod[i];
      if (cur_pow < 0) {
        cur_pow = 0;
      }
    }
    // pow(x, 0) == 1, so skip pow.
    if (cur_pow == 0.0) {
      for (int y = 0; y < quant_field.ysize(); ++y) {
        const float* const PIK_RESTRICT row_dist = tile_distmap.Row(y);
        float* const PIK_RESTRICT row_q = quant_field.Row(y);
        for (int x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff >= 1.0f) {
            row_q[x] *= diff;
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    } else {
      for (int y = 0; y < quant_field.ysize(); ++y) {
        const float* const PIK_RESTRICT row_dist = tile_distmap.Row(y);
        float* const PIK_RESTRICT row_q = quant_field.Row(y);
        for (int x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff < 1.0f) {
            row_q[x] *= pow(diff, cur_pow);
          } else {
            row_q[x] *= diff;
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    }
  }
  quantizer->SetQuantField(initial_quant_dc, QuantField(quant_field));
}

void FindBestQuantizationHQ(
    const Image3F& opsin_orig, const Image3F& opsin,
    const CompressParams& cparams, const FrameHeader& frame_header,
    const GroupHeader& header, float butteraugli_target,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& sigma_lut_ids,
    ImageF& quant_field, Quantizer* quantizer,
    const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    PikInfo* aux_out, MultipassManager* multipass_manager, double rescale) {
  const float intensity_multiplier = cparams.GetIntensityMultiplier();
  const float intensity_multiplier3 = std::cbrt(intensity_multiplier);
  ButteraugliComparator comparator(opsin_orig, cparams.hf_asymmetry,
                                   intensity_multiplier);
  AdjustQuantField(ac_strategy, &quant_field);
  ImageF best_quant_field = CopyImage(quant_field);
  float best_butteraugli = 1000.0f;
  ImageF tile_distmap;
  static const int kMaxOuterIters = 2;
  int outer_iter = 0;
  int butteraugli_iter = 0;
  int search_radius = 0;
  float quant_ceil = 5.0f;
  float quant_dc = intensity_multiplier3 * 1.2f;
  float best_quant_dc = quant_dc;
  int num_stalling_iters = 0;
  int max_iters = cparams.max_butteraugli_iters_guetzli_mode;

  for (;;) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }
    float qmin, qmax;
    ImageMinMax(quant_field, &qmin, &qmax);
    ++butteraugli_iter;
    if (quantizer->SetQuantField(quant_dc, QuantField(quant_field))) {
      Image3F linear = RoundtripImage(
          cparams, frame_header, header, opsin_orig, opsin, cmap,
          block_dictionary, ac_strategy, sigma_lut_ids, *quantizer,
          dequant_control_field, dequant_map, multipass_manager);
      comparator.Compare(linear);
      bool best_quant_updated = false;
      if (comparator.distance() <= best_butteraugli) {
        best_quant_field = CopyImage(quant_field);
        best_butteraugli = std::max(comparator.distance(), butteraugli_target);
        best_quant_updated = true;
        best_quant_dc = quant_dc;
        num_stalling_iters = 0;
      } else if (outer_iter == 0) {
        ++num_stalling_iters;
      }
      tile_distmap = TileDistMap(comparator.distmap(), 8, 0, ac_strategy);
      if (WantDebugOutput(aux_out)) {
        DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
      }
      if (aux_out) {
        ++aux_out->num_butteraugli_iters;
      }
      if (FLAGS_log_search_state) {
        float minval, maxval;
        ImageMinMax(quant_field, &minval, &maxval);
        printf("\nButteraugli iter: %d/%d%s\n", butteraugli_iter, max_iters,
               best_quant_updated ? " (*)" : "");
        printf("Butteraugli distance: %f\n", comparator.distance());
        printf(
            "quant range: %f ... %f  DC quant: "
            "%f\n",
            minval, maxval, quant_dc);
        printf("search radius: %d\n", search_radius);
        if (FLAGS_dump_quant_state) {
          quantizer->DumpQuantizationMap();
        }
      }
    }
    if (butteraugli_iter >= max_iters) {
      break;
    }
    bool changed = false;
    while (!changed && comparator.distance() > butteraugli_target) {
      for (int radius = 0; radius <= search_radius && !changed; ++radius) {
        ImageF dist_to_peak_map =
            DistToPeakMap(tile_distmap, butteraugli_target, radius, 0.0);
        for (int y = 0; y < quant_field.ysize(); ++y) {
          float* const PIK_RESTRICT row_q = quant_field.Row(y);
          const float* const PIK_RESTRICT row_dist = dist_to_peak_map.Row(y);
          for (int x = 0; x < quant_field.xsize(); ++x) {
            if (row_dist[x] >= 0.0f) {
              static const float kAdjSpeed[kMaxOuterIters] = {0.1f, 0.04f};
              const float factor =
                  kAdjSpeed[outer_iter] * tile_distmap.Row(y)[x];
              if (AdjustQuantVal(&row_q[x], row_dist[x], factor, quant_ceil)) {
                changed = true;
              }
            }
          }
        }
      }
      if (!changed || num_stalling_iters >= 3) {
        // Try to extend the search parameters.
        if ((search_radius < 4) &&
            (qmax < 0.99f * quant_ceil || quant_ceil >= 3.0f + search_radius)) {
          ++search_radius;
          continue;
        }
        if (quant_dc < 0.4f * quant_ceil - 0.8f) {
          quant_dc += 0.2f;
          changed = true;
          continue;
        }
        if (quant_ceil < 8.0f) {
          quant_ceil += 0.5f;
          continue;
        }
        break;
      }
    }
    if (!changed) {
      if (++outer_iter == kMaxOuterIters) break;
      static const float kQuantScale = 0.75f;
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          quant_field.Row(y)[x] *= kQuantScale;
        }
      }
      num_stalling_iters = 0;
    }
  }
  quantizer->SetQuantField(best_quant_dc, QuantField(best_quant_field));
}

ImageF AdaptiveQuantizationMap(const Image3F& img, const ImageF& img_ac,
                               const CompressParams& cparams) {
  PROFILER_ZONE("aq AdaptiveQuantMap");
  static const int kResolution = 8;
  const size_t out_xsize = (img.xsize() + kResolution - 1) / kResolution;
  const size_t out_ysize = (img.ysize() + kResolution - 1) / kResolution;
  if (img.xsize() <= 1) {
    ImageF out(1, out_ysize);
    FillImage(1.0f, &out);
    return out;
  }
  if (img.ysize() <= 1) {
    ImageF out(out_xsize, 1);
    FillImage(1.0f, &out);
    return out;
  }
  static const float kSigma = 8.2553856725566153;
  static const int kRadius = static_cast<int>(2 * kSigma + 0.5f);
  std::vector<float> kernel = GaussianKernel(kRadius, kSigma);
  static const float kDiffCutoff = 0.11883287948847132;
  ImageF out = DiffPrecompute(img, kDiffCutoff);
  out = Expand(out, kResolution * out_xsize, kResolution * out_ysize);
  out = ConvolveAndSample(out, kernel, kResolution);
  out = ComputeMask(out);
//  DctModulation(img_ac, &out);
//  RangeModulation(img_ac, &out);
//  HfModulation(img_ac, &out);
  Exp(&out);
  return out;
}

// TODO(veluca): remove or use pool.
ImageF IntensityAcEstimate(const ImageF& image, float multiplier,
                           ThreadPool* pool) {
  constexpr size_t N = kBlockDim;
  std::vector<float> blur = DCfiedGaussianKernel<N>(5.5);
  ImageF retval = Convolve(image, blur);
  for (size_t y = 0; y < retval.ysize(); y++) {
    float* PIK_RESTRICT retval_row = retval.Row(y);
    const float* PIK_RESTRICT image_row = image.ConstRow(y);
    for (size_t x = 0; x < retval.xsize(); ++x) {
      retval_row[x] = multiplier * (image_row[x] - retval_row[x]);
    }
  }
  return retval;
}

}  // namespace

float InitialQuantDC(float butteraugli_target, float intensity_multiplier) {
  const float intensity_multiplier3 = std::cbrt(intensity_multiplier);
  const float butteraugli_target_dc =
      std::min<float>(butteraugli_target, pow(butteraugli_target, kDcQuantPow));
  return intensity_multiplier3 * kDcQuant / butteraugli_target_dc;
}

ImageF InitialQuantField(double butteraugli_target, double intensity_multiplier,
                         const Image3F& opsin_orig,
                         const CompressParams& cparams, ThreadPool* pool,
                         double rescale) {
  const float intensity_multiplier3 = std::cbrt(intensity_multiplier);
  const float quant_ac = intensity_multiplier3 * kAcQuant / butteraugli_target;
  ImageF intensity_ac =
      IntensityAcEstimate(opsin_orig.Plane(1), intensity_multiplier3, pool);
  ImageF quant_field =
      ScaleImage(quant_ac * (float)rescale,
                 AdaptiveQuantizationMap(opsin_orig, intensity_ac, cparams));
  return quant_field;
}

std::shared_ptr<Quantizer> FindBestQuantizer(
    const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const FrameHeader& frame_header, const GroupHeader& header,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
    const DequantMatrices* dequant, const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    ImageF& quant_field, PikInfo* aux_out,
    MultipassManager* multipass_manager, double rescale) {
  std::shared_ptr<Quantizer> quantizer =
      std::make_shared<Quantizer>(dequant, xsize_blocks, ysize_blocks);
  const float intensity_multiplier = cparams.GetIntensityMultiplier();
  if (cparams.fast_mode) {
    PROFILER_ZONE("enc fast quant");
    const float butteraugli_target = cparams.butteraugli_distance;
    const float quant_dc =
        InitialQuantDC(butteraugli_target, intensity_multiplier);
    Rect full(opsin_orig);
    // TODO(veluca): warn if uniform_quant is set - or honor it
    AdjustQuantField(ac_strategy, &quant_field);
    quantizer->SetQuantField(quant_dc, QuantField(quant_field));
  } else if (cparams.uniform_quant > 0.0) {
    PROFILER_ZONE("enc SetQuant");
    quantizer->SetQuant(cparams.uniform_quant * rescale);
  } else {
    // Normal PIK encoding to a butteraugli score.
    PROFILER_ZONE("enc find best2");
    if (cparams.guetzli_mode) {
      FindBestQuantizationHQ(opsin_orig, opsin, cparams, frame_header, header,
                             cparams.butteraugli_distance, cmap,
                             block_dictionary, ac_strategy, ar_sigma_lut_ids,
                             quant_field, quantizer.get(),
                             dequant_control_field, dequant_map, aux_out,
                             multipass_manager, rescale);
    } else {
      FindBestQuantization(opsin_orig, opsin, cparams, frame_header, header,
                           cparams.butteraugli_distance, cmap, block_dictionary,
                           ac_strategy, ar_sigma_lut_ids, quant_field,
                           quantizer.get(), dequant_control_field, dequant_map,
                           aux_out, multipass_manager, rescale);
    }
  }
  return quantizer;
}

std::shared_ptr<Quantizer> FindBestQuantizerAvg(float avg, float absavg,
    const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const FrameHeader& frame_header, const GroupHeader& header,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
    const DequantMatrices* dequant, const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    ImageF& quant_field, PikInfo* aux_out,
    MultipassManager* multipass_manager, double rescale) {
  std::shared_ptr<Quantizer> quantizer =
      std::make_shared<Quantizer>(dequant, xsize_blocks, ysize_blocks);
  const float intensity_multiplier = cparams.GetIntensityMultiplier();
  if (cparams.fast_mode) {
    PROFILER_ZONE("enc fast quant");
    const float butteraugli_target = cparams.butteraugli_distance;
    const float quant_dc =
        InitialQuantDC(butteraugli_target, intensity_multiplier);
    Rect full(opsin_orig);
    // TODO(veluca): warn if uniform_quant is set - or honor it
    ImageF qfOrigin = CopyImage(quant_field);
    AdjustQuantField(ac_strategy, &quant_field);
    quantizer->SetQuantFieldOR(avg, absavg, quant_dc, QuantField(quant_field), qfOrigin);
  } else if (cparams.uniform_quant > 0.0) {
    PROFILER_ZONE("enc SetQuant");
    quantizer->SetQuant(cparams.uniform_quant * rescale);
  } else {
    // Normal PIK encoding to a butteraugli score.
    PROFILER_ZONE("enc find best2");
    if (cparams.guetzli_mode) {
      FindBestQuantizationHQ(opsin_orig, opsin, cparams, frame_header, header,
                             cparams.butteraugli_distance, cmap,
                             block_dictionary, ac_strategy, ar_sigma_lut_ids,
                             quant_field, quantizer.get(),
                             dequant_control_field, dequant_map, aux_out,
                             multipass_manager, rescale);
    } else {
      FindBestQuantization(opsin_orig, opsin, cparams, frame_header, header,
                           cparams.butteraugli_distance, cmap, block_dictionary,
                           ac_strategy, ar_sigma_lut_ids, quant_field,
                           quantizer.get(), dequant_control_field, dequant_map,
                           aux_out, multipass_manager, rescale);
    }
  }
  return quantizer;
}

}  // namespace pik
