// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/color_correlation.h"
#include "pik/huffman_decode.h"
#include "pik/huffman_encode.h"
#include "pik/write_bits.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/common.h"
#include "pik/dct_util.h"
#include "pik/profiler.h"
#include "pik/quantizer.h"

namespace pik {

namespace {
template <typename V, typename R>
inline void FindIndexOfSumMaximum(const V* array, const size_t len, R* idx,
                                  V* sum) {
  PIK_ASSERT(len > 0);
  V maxval = 0;
  V val = 0;
  R maxidx = 0;
  for (size_t i = 1; i < len; ++i) {
    val += array[i];
    if (val > maxval) {
      maxval = val;
      maxidx = i;
    }
  }
  *idx = maxidx;
  *sum = maxval;
}

template <int MAIN_CHANNEL, int SIDE_CHANNEL, int SCALE, int OFFSET>
void FindBestCorrelation(const Image3F& dct, ImageI* PIK_RESTRICT map,
                         ImageF* PIK_RESTRICT tmp_map, int* PIK_RESTRICT dc,
                         float acceptance, const DequantMatrices& dequant) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  constexpr float kScale = SCALE;
  constexpr float kZeroThresh = kScale * kZeroBiasDefault[SIDE_CHANNEL];
  // Always use DCT8 quantization values for DC.
  const float* const PIK_RESTRICT kDequantMatrix =
      dequant.Matrix(0, kQuantKindDCT8, SIDE_CHANNEL);
  float qm[block_size];
  for (int k = 0; k < block_size; ++k) {
    qm[k] = 1.0f / kDequantMatrix[k];
  }
  int32_t d_num_zeros_global[256] = {0};
  for (int ty = 0; ty < map->ysize(); ++ty) {
    int* PIK_RESTRICT row_out = map->Row(ty);
    float* PIK_RESTRICT row_tmp_out = tmp_map->Row(ty);
    for (int tx = 0; tx < map->xsize(); ++tx) {
      const int y0 = ty * kColorTileDimInBlocks;
      const int x0 = tx * kColorTileDimInBlocks * block_size;
      const int y1 = std::min<int>(y0 + kColorTileDimInBlocks, dct.ysize());
      const int x1 =
          std::min<int>(x0 + kColorTileDimInBlocks * block_size, dct.xsize());
      int32_t d_num_zeros[257] = {0};
      for (size_t y = y0; y < y1; ++y) {
        const float* const PIK_RESTRICT row_m =
            dct.ConstPlaneRow(MAIN_CHANNEL, y);
        const float* const PIK_RESTRICT row_s =
            dct.ConstPlaneRow(SIDE_CHANNEL, y);
        for (size_t x = x0; x < x1; ++x) {
          if (x % block_size == 0) continue;
          const float scaled_m = row_m[x] * qm[x % block_size];
          const float scaled_s =
              kScale * row_s[x] * qm[x % block_size] + OFFSET * scaled_m;
          // Increment num_zeros[idx] if
          //   std::abs(scaled_s - (idx - OFFSET) *
          //   scaled_m) < kZeroThresh
          if (std::abs(scaled_m) < 1e-8) {
            // Range is too narrow, all-or-nothing
            // strategy should be OK.
            if (std::abs(scaled_s) < kZeroThresh) {
              d_num_zeros[0]++;
            }
          } else {
            float from;
            float to;
            if (scaled_m > 0) {
              from = (scaled_s - kZeroThresh) / scaled_m;
              to = (scaled_s + kZeroThresh) / scaled_m;
            } else {
              from = (scaled_s + kZeroThresh) / scaled_m;
              to = (scaled_s - kZeroThresh) / scaled_m;
            }
            if (from < 0.0f) {
              from = 0.0f;
            }
            if (to > 255.0f) {
              to = 255.0f;
            }
            // Instead of clamping the both values
            // we just check that range is sane.
            if (from <= to) {
              d_num_zeros[(int)std::ceil(from)]++;
              d_num_zeros[(int)std::floor(to + 1)]--;
            }
          }
        }
      }
      int best = 0;
      int32_t best_sum = 0;
      FindIndexOfSumMaximum(d_num_zeros, 256, &best, &best_sum);
      for (size_t i = 0; i < 256; ++i) {
        d_num_zeros_global[i] += d_num_zeros[i];
      }
      row_out[tx] = best;
      row_tmp_out[tx] = (float)best_sum / ((x1 - x0) * (y1 - y0));
    }
  }

  int global_best = 0;
  int32_t global_sum = 0;
  FindIndexOfSumMaximum(d_num_zeros_global, 256, &global_best, &global_sum);
  float global_normalized_sum = (float)global_sum / (dct.xsize() * dct.ysize());
  float normalized_acceptance =
      acceptance * kColorTileDimInBlocks * kColorTileDimInBlocks * block_size;
  for (int ty = 0; ty < map->ysize(); ++ty) {
    int* PIK_RESTRICT row_out = map->Row(ty);
    float* PIK_RESTRICT row_tmp_out = tmp_map->Row(ty);
    for (int tx = 0; tx < map->xsize(); ++tx) {
      // Revert to the global factor used for dc if
      // the number of zeros is almost the same.
      if (row_tmp_out[tx] <= global_normalized_sum + normalized_acceptance) {
        row_out[tx] = global_best;
      }
    }
  }
  *dc = global_best;
}

}  // namespace

// "y_plane" may refer to plane#1 of "coeffs"; it is also organized in the
// block layout (consecutive block coefficient `pixels').
// Class Dequant applies color correlation maps back.
SIMD_ATTR void UnapplyColorCorrelationAC(const ColorCorrelationMap& cmap,
                                         const Rect& cmap_rect,
                                         const ImageF& y_plane,
                                         Image3F* coeffs) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const SIMD_FULL(float) d;

  const size_t xsize_blocks = coeffs->xsize() / block_size;
  const size_t ysize_blocks = coeffs->ysize();
  for (size_t y = 0; y < ysize_blocks; ++y) {
    size_t ty = y / kColorTileDimInBlocks;
    const int* PIK_RESTRICT row_ytob = cmap_rect.ConstRow(cmap.ytob_map, ty);
    const int* PIK_RESTRICT row_ytox = cmap_rect.ConstRow(cmap.ytox_map, ty);

    for (size_t x = 0; x < xsize_blocks; ++x) {
      size_t tx = x / kColorTileDimInBlocks;
      const float* PIK_RESTRICT row_y = y_plane.Row(y) + x * block_size;
      float* PIK_RESTRICT row_x = coeffs->PlaneRow(0, y) + x * block_size;
      float* PIK_RESTRICT row_b = coeffs->PlaneRow(2, y) + x * block_size;
      const auto ytob = set1(d, ColorCorrelationMap::YtoB(1.0f, row_ytob[tx]));
      const auto ytox = set1(d, ColorCorrelationMap::YtoX(1.0f, row_ytox[tx]));
      for (size_t k = 0; k < block_size; k += d.N) {
        const auto in_y = load(d, row_y + k);
        const auto in_b = load(d, row_b + k);
        const auto in_x = load(d, row_x + k);
        const auto out_b = in_b - ytob * in_y;
        const auto out_x = in_x - ytox * in_y;
        store(out_b, d, row_b + k);
        store(out_x, d, row_x + k);
      }
    }
  }
}

template <bool decode>
SIMD_ATTR void ApplyColorCorrelationDC(const ColorCorrelationMap& cmap,
                                       const ImageF& y_plane_dc,
                                       Image3F* coeffs_dc) {
  const SIMD_FULL(float) d;
  const size_t xsize_blocks = coeffs_dc->xsize();
  const size_t ysize_blocks = coeffs_dc->ysize();

  const auto ytob = set1(d, ColorCorrelationMap::YtoB(1.0f, cmap.ytob_dc));
  const auto ytox = set1(d, ColorCorrelationMap::YtoX(1.0f, cmap.ytox_dc));

  //std::cout<<"std ytox="<<cmap.ytox_dc<<std::endl;
  //std::cout<<"std ytob="<<cmap.ytob_dc<<std::endl;
  //std::cout<<"std YtoX="<<ytox.raw<<std::endl;
  //std::cout<<"std YtoB="<<ytob.raw<<std::endl;

  for (size_t y = 0; y < ysize_blocks; ++y) {
    const float* PIK_RESTRICT row_y = y_plane_dc.Row(y);
    float* PIK_RESTRICT row_x = coeffs_dc->PlaneRow(0, y);
    float* PIK_RESTRICT row_b = coeffs_dc->PlaneRow(2, y);

    for (size_t x = 0; x < xsize_blocks; x += d.N) {

      const auto in_y = load(d, row_y + x);
      const auto in_b = load(d, row_b + x);
      const auto in_x = load(d, row_x + x);

      //std::cout<<"cor_pre: y="<<y<<" x="<<x<<" X="<<(float)row_x[x]<<" Y="<<(float)row_y[x]<<" B="<<(float)row_b[x]<<std::endl;

      const auto out_b = decode ? in_b + ytob * in_y : in_b - ytob * in_y;
      const auto out_x = decode ? in_x + ytox * in_y : in_x - ytox * in_y;
      store(out_b, d, row_b + x);
      store(out_x, d, row_x + x);

      //std::cout<<"cor_post: y="<<y<<" x="<<x<<" X="<<(float)row_x[x]<<" Y="<<(float)row_y[x]<<" B="<<(float)row_b[x]<<std::endl;
    }
  }
}

template void ApplyColorCorrelationDC<true>(const ColorCorrelationMap&,
                                            const ImageF&, Image3F*);

template void ApplyColorCorrelationDC<false>(const ColorCorrelationMap&,
                                             const ImageF&, Image3F*);

void FindBestColorCorrelationMap(const Image3F& opsin,
                                 const DequantMatrices& dequant,
                                 ColorCorrelationMap* cmap) {
  PROFILER_ZONE("enc YTo* correlation");

  constexpr int block_size = kBlockDim * kBlockDim;
  const size_t xsize_blocks = opsin.xsize() / kBlockDim;
  const size_t ysize_blocks = opsin.ysize() / kBlockDim;
  Image3F dct(xsize_blocks * block_size, ysize_blocks);
  TransposedScaledDCT(opsin, &dct);

  ImageF tmp(DivCeil(opsin.xsize(), kColorTileDim),
             DivCeil(opsin.ysize(), kColorTileDim));

  // These two coefficients are eligible for optimization.
  // Perhaps, they also could be made quality-dependent.
  // Prefer global until 25% more (full) tile coefficients become zero.
  float y_to_b_acceptance = 0.25f;
  // Prefer local until 62.5% less (full) tile coefficients become zero.
  float y_to_x_acceptance = -0.625f;

  FindBestCorrelation</* from Y */ 1, /* to B */ 2, kColorFactorB,
                      kColorOffsetB>(dct, &cmap->ytob_map, &tmp, &cmap->ytob_dc,
                                     y_to_b_acceptance, dequant);
  FindBestCorrelation</* from Y */ 1, /* to X */ 0, kColorFactorX,
                      kColorOffsetX>(dct, &cmap->ytox_map, &tmp, &cmap->ytox_dc,
                                     y_to_x_acceptance, dequant);
}

bool DecodeColorMap(BitReader* PIK_RESTRICT br, ImageI* PIK_RESTRICT ac_map,
                    int* PIK_RESTRICT dc_val) {
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(br)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  br->FillBitBuffer();
  *dc_val = decoder.ReadSymbol(entropy, br);
  for (size_t y = 0; y < ac_map->ysize(); ++y) {
    int* PIK_RESTRICT row = ac_map->Row(y);
    for (size_t x = 0; x < ac_map->xsize(); ++x) {
      br->FillBitBuffer();
      row[x] = decoder.ReadSymbol(entropy, br);
    }
  }
  PIK_RETURN_IF_ERROR(br->JumpToByteBoundary());
  return true;
}

std::string EncodeColorMap(const ImageI& ac_map, const Rect& rect,
                           const int dc_val, PikImageSizeInfo* info) {
  PIK_ASSERT(rect.IsInside(ac_map));
  const size_t max_out_size = rect.xsize() * rect.ysize() + 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  std::vector<uint32_t> histogram(256);
  ++histogram[dc_val];
  for (int y = 0; y < rect.ysize(); ++y) {
    for (int x = 0; x < rect.xsize(); ++x) {
      ++histogram[rect.ConstRow(ac_map, y)[x]];
    }
  }
  std::vector<uint8_t> bit_depths(256);
  std::vector<uint16_t> bit_codes(256);
  BuildAndStoreHuffmanTree(histogram.data(), histogram.size(),
                           bit_depths.data(), bit_codes.data(), &storage_ix,
                           storage);
  const size_t histo_bits = storage_ix;
  WriteBits(bit_depths[dc_val], bit_codes[dc_val], &storage_ix, storage);
  for (int y = 0; y < rect.ysize(); ++y) {
    const int* PIK_RESTRICT row = rect.ConstRow(ac_map, y);
    for (int x = 0; x < rect.xsize(); ++x) {
      WriteBits(bit_depths[row[x]], bit_codes[row[x]], &storage_ix, storage);
    }
  }
  WriteZeroesToByteBoundary(&storage_ix, storage);
  PIK_ASSERT((storage_ix >> 3) <= output.size());
  output.resize(storage_ix >> 3);
  if (info) {
    info->histogram_size += histo_bits >> 3;
    info->entropy_coded_bits += storage_ix - histo_bits;
    info->total_size += output.size();
  }
  return output;
}

}  // namespace pik
