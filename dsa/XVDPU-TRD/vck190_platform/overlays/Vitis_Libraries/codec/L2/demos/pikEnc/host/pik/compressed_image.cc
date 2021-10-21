// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/compressed_image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/ac_predictions.h"
#include "pik/ac_strategy.h"
#include "pik/adaptive_reconstruction.h"
#include "pik/ans_decode.h"
#include "pik/bilinear_transform.h"
#include "pik/block.h"
#include "pik/butteraugli_distance.h"
#include "pik/color_correlation.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/compressed_dc.h"
#include "pik/compressed_image_fwd.h"
#include "pik/convolve.h"
#include "pik/dc_predictor.h"
#include "pik/dct.h"
#include "pik/dct_util.h"
#include "pik/deconvolve.h"
#include "pik/entropy_coder.h"
#include "pik/epf.h"
#include "pik/fields.h"
#include "pik/gaborish.h"
#include "pik/gauss_blur.h"
#include "pik/gradient_map.h"
#include "pik/headers.h"
#include "pik/huffman_decode.h"
#include "pik/huffman_encode.h"
#include "pik/image.h"
#include "pik/image_ops.h"
#include "pik/lossless16.h"
#include "pik/lossless8.h"
#include "pik/opsin_image.h"
#include "pik/opsin_inverse.h"
#include "pik/opsin_params.h"
#include "pik/pik_params.h"
#include "pik/profiler.h"
#include "pik/quantizer.h"
#include "pik/resample.h"
#include "pik/resize.h"
#include "pik/simd/simd.h"
#include "pik/status.h"
#include "pik/upscaler.h"

#define USE_K2_DATA true

namespace pik {

namespace {

void ZeroDcValues(Image3F *image, const AcStrategyImage &ac_strategy) {
  const constexpr size_t N = kBlockDim;
  const size_t xsize_blocks = image->xsize() / (N * N);
  const size_t ysize_blocks = image->ysize();
  for (size_t c = 0; c < image->kNumPlanes; c++) {
    for (size_t by = 0; by < ysize_blocks; by++) {
      AcStrategyRow acs_row = ac_strategy.ConstRow(by);
      float *PIK_RESTRICT stored_values = image->PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; bx++) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock())
          continue;
        for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
          for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
            stored_values[bx * N * N + y * acs.covered_blocks_x() * N + x] = 0;
          }
        }
      }
    }
  }
}

} // namespace

constexpr float kIdentityAvgParam = 0.25;

// This struct allow to remove the X and B channels of XYB images, and
// reconstruct them again from only the Y channel, when the image is grayscale.
struct GrayXyb {
  static const constexpr int kM = 16; // Amount of line pieces.

  GrayXyb() { Compute(); }

  void YToXyb(float y, float *x, float *b) const {
    int i = (int)((y - ysub) * ymul * kM);
    i = std::min(std::max(0, i), kM - 1);
    *x = y * y_to_x_slope[i] + y_to_x_constant[i];
    *b = y * y_to_b_slope[i] + y_to_b_constant[i];
  }

  void RemoveXB(Image3F *image) const {
    for (size_t y = 0; y < image->ysize(); ++y) {
      float *PIK_RESTRICT row_x = image->PlaneRow(0, y);
      float *PIK_RESTRICT row_b = image->PlaneRow(2, y);
      for (size_t x = 0; x < image->xsize(); x++) {
        row_x[x] = 0;
        row_b[x] = 0;
      }
    }
  }

  void RestoreXB(Image3F *image) const {
    for (size_t y = 0; y < image->ysize(); ++y) {
      const float *PIK_RESTRICT row_y = image->PlaneRow(1, y);
      float *PIK_RESTRICT row_x = image->PlaneRow(0, y);
      float *PIK_RESTRICT row_b = image->PlaneRow(2, y);
      for (size_t x = 0; x < image->xsize(); x++) {
        YToXyb(row_y[x], &row_x[x], &row_b[x]);
      }
    }
  }

private:
  void Compute() {
    static const int kN = 1024;
    std::vector<float> x(kN);
    std::vector<float> y(kN);
    std::vector<float> z(kN);
    for (int i = 0; i < kN; i++) {
      float gray = (float)(256.0f * i / kN);
      LinearToXyb(gray, gray, gray, &x[i], &y[i], &z[i]);
    }

    float min = y[0];
    float max = y[kN - 1];
    int m = 0;
    int border[kM + 1];
    for (int i = 0; i < kN; i++) {
      if (y[i] >= y[0] + (max - min) * m / kM) {
        border[m] = i;
        m++;
      }
    }
    border[kM] = kN;

    ysub = min;
    ymul = 1.0 / (max - min);

    for (int i = 0; i < kM; i++) {
      LinearRegression(y.data() + border[i], x.data() + border[i],
                       border[i + 1] - border[i], &y_to_x_constant[i],
                       &y_to_x_slope[i]);
      LinearRegression(y.data() + border[i], z.data() + border[i],
                       border[i + 1] - border[i], &y_to_b_constant[i],
                       &y_to_b_slope[i]);
    }
  }

  // finds a and b such that y ~= b*x + a
  void LinearRegression(const float *x, const float *y, size_t size, double *a,
                        double *b) {
    double mx = 0, my = 0;   // mean
    double mx2 = 0, my2 = 0; // second moment
    double mxy = 0;
    for (size_t i = 0; i < size; i++) {
      double inv = 1.0 / (i + 1);

      double dx = x[i] - mx;
      double xn = dx * inv;
      mx += xn;
      mx2 += dx * xn * i;

      double dy = y[i] - my;
      double yn = dy * inv;
      my += yn;
      my2 += dy * yn * i;

      mxy += i * xn * yn - mxy * inv;
    }

    double sx = std::sqrt(mx2 / (size - 1));
    double sy = std::sqrt(my2 / (size - 1));

    double sumxy = mxy * size + my * mx * size;
    double r = (sumxy - size * mx * my) / ((size - 1.0) * sx * sy);

    *b = r * sy / sx;
    *a = my - *b * mx;
  }

  double y_to_x_slope[kM];
  double y_to_x_constant[kM];
  double y_to_b_slope[kM];
  double y_to_b_constant[kM];

  double ysub;
  double ymul;
};

// Gets the singleton GrayXyb instance.
static const GrayXyb *GetGrayXyb() {
  static const GrayXyb *kGrayXyb = new GrayXyb;
  return kGrayXyb;
}

SIMD_ATTR void InitializeFrameEncCache(
    const FrameHeader &frame_header, const Image3F &opsin_full,
    const AcStrategyImage &ac_strategy, const Quantizer &quantizer,
    const ColorCorrelationMap &cmap, const BlockDictionary &dictionary,
    FrameEncCache *frame_enc_cache, PikInfo *aux_out) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  frame_enc_cache->ac_strategy = ac_strategy.Copy();
  frame_enc_cache->use_gradient =
      frame_header.flags & FrameHeader::kGradientMap;
  frame_enc_cache->grayscale_opt =
      frame_header.flags & FrameHeader::kGrayscaleOpt;
  const size_t xsize_blocks = opsin_full.xsize() / N;
  const size_t ysize_blocks = opsin_full.ysize() / N;

  Image3F opsin = CopyImage(opsin_full);
  dictionary.SubtractFrom(&opsin);

  ApplyReverseBilinear(&opsin);

  frame_enc_cache->coeffs = Image3F(xsize_blocks * kDCTBlockSize, ysize_blocks);
  Image3F dc = Image3F(xsize_blocks, ysize_blocks);

  for (int by = 0; by < ysize_blocks; ++by) {
    for (int c = 0; c < 3; ++c) {
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        AcStrategy acs = ac_strategy.ConstRow(by)[bx];
        acs.TransformFromPixels(
            opsin.ConstPlaneRow(c, by * N) + bx * N, opsin.PixelsPerRow(),
            frame_enc_cache->coeffs.PlaneRow(c, by) + bx * kDCTBlockSize,
            frame_enc_cache->coeffs.PixelsPerRow());
      }
    }
  }

  /*
   for (int c = 0; c < 3; ++c) {
     for (int by = 0; by < ysize_blocks; ++by) {
       for (size_t bx = 0; bx < xsize_blocks; ++bx) {
           float* tmp=frame_enc_cache->coeffs.PlaneRow(c, by) + bx*64;
           for (int k=0;k<64;k++)
           std::cout<<"std_dct: c="<<c<<" by="<<by<<" bx="<<bx<<" k="<<k<<"
   "<<tmp[k]<<std::endl;
       }
     }
   }
   */

  for (int by = 0; by < ysize_blocks; ++by) {
    for (int c = 0; c < 3; ++c) {
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        AcStrategy acs = ac_strategy.ConstRow(by)[bx];
        acs.DCFromLowestFrequencies(
            frame_enc_cache->coeffs.ConstPlaneRow(c, by) + bx * kDCTBlockSize,
            frame_enc_cache->coeffs.PixelsPerRow(), dc.PlaneRow(c, by) + bx,
            dc.PixelsPerRow());
      }
    }
  }

  if (aux_out != nullptr) {
    aux_out->InspectImage3F("compressed_image:InitializeFrameEncCache:dc", dc);
  }

  constexpr int cY = 1; // Y color channel.

  
  std::cout<<"std_dc_float:"<<std::endl;
  for(int c=0;c<3;c++){
          for(int y=0;y<ysize_blocks;y++){
                  for(int x=0;x<xsize_blocks;x++){
                          const float* PIK_RESTRICT row=dc.ConstPlaneRow(c, y);
                          std::cout<<row[x]<<",";
                  }
          std::cout<<std::endl;
          }
      std::cout<<std::endl;
  }
  
  {
    ImageF dec_dc_Y = QuantizeRoundtripDC(quantizer, cY, dc.Plane(cY));

    ApplyColorCorrelationDC</*decode=*/false>(cmap, dec_dc_Y, &dc);

    
    std::cout<<"std_dc_apply_false:";
      for(int c=0;c<3;c++){
              for(int y=0;y<ysize_blocks;y++){
                      for(int x=0;x<xsize_blocks;x++){
                              const float* PIK_RESTRICT row=dc.ConstPlaneRow(c,
    y);
                              std::cout<<row[x]<<",";
                      }
              std::cout<<std::endl;
              }
          std::cout<<std::endl;
      }
    
    frame_enc_cache->dc = QuantizeCoeffsDC(dc, quantizer);

    std::cout << "std_dc_x:";
      for (size_t y = 0; y < frame_enc_cache->dc.ysize(); y++) {
        const int16_t *PIK_RESTRICT row_in =
            frame_enc_cache->dc.ConstPlaneRow(0, y);
        for (size_t x = 0; x < frame_enc_cache->dc.xsize(); x++) {
          std::cout << row_in[x] << ",";
        }
        std::cout << std::endl;
      }

      std::cout << "std_dc_y:";
      for (size_t y = 0; y < frame_enc_cache->dc.ysize(); y++) {
        const int16_t *PIK_RESTRICT row_in =
            frame_enc_cache->dc.ConstPlaneRow(1, y);
        for (size_t x = 0; x < frame_enc_cache->dc.xsize(); x++) {
          std::cout << row_in[x] << ",";
        }
        std::cout << std::endl;
      }

      std::cout << "std_dc_b:";
      for (size_t y = 0; y < frame_enc_cache->dc.ysize(); y++) {
        const int16_t *PIK_RESTRICT row_in =
            frame_enc_cache->dc.ConstPlaneRow(2, y);
        for (size_t x = 0; x < frame_enc_cache->dc.xsize(); x++) {
          std::cout << row_in[x] << ",";
        }
        std::cout << std::endl;
      }

    frame_enc_cache->dc_dec =
        Image3F(frame_enc_cache->dc.xsize(), frame_enc_cache->dc.ysize());

    for (size_t c = 0; c < 3; c++) {
      const float mul = quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0] *
                        quantizer.inv_quant_dc();
      for (size_t y = 0; y < frame_enc_cache->dc.ysize(); y++) {
        const int16_t *PIK_RESTRICT row_in =
            frame_enc_cache->dc.ConstPlaneRow(c, y);
        float *PIK_RESTRICT row_out = frame_enc_cache->dc_dec.PlaneRow(c, y);
        for (size_t x = 0; x < frame_enc_cache->dc.xsize(); x++) {
          row_out[x] = row_in[x] * mul;
        }
      }
    }

    // std::cout<<"mul_x="<<quantizer.DequantMatrix(0, kQuantKindDCT8, 0)[0] *
    // quantizer.inv_quant_dc()<<std::endl;
    // std::cout<<"mul_y="<<quantizer.DequantMatrix(0, kQuantKindDCT8, 1)[0] *
    // quantizer.inv_quant_dc()<<std::endl;
    // std::cout<<"mul_b="<<quantizer.DequantMatrix(0, kQuantKindDCT8, 2)[0] *
    // quantizer.inv_quant_dc()<<std::endl;

    ApplyColorCorrelationDC</*decode=*/true>(cmap, dec_dc_Y,
                                             &frame_enc_cache->dc_dec);

    std::cout<<"std_dc_apply_true:";
      for(int c=0;c<3;c++){
              for(int y=0;y<ysize_blocks;y++){
                      for(int x=0;x<xsize_blocks;x++){
                              const float* PIK_RESTRICT row=frame_enc_cache->dc_dec.PlaneRow(c, y);
                              std::cout<<row[x]<<",";
                      }
              std::cout<<std::endl;
              }
          std::cout<<std::endl;
      }

    AdaptiveDCReconstruction(frame_enc_cache->dc_dec, quantizer);

    std::cout<<"std_AdaptiveDCReconstruction:";
      for(int c=0;c<3;c++){
              for(int y=0;y<ysize_blocks;y++){
                      for(int x=0;x<xsize_blocks;x++){
                              const float* PIK_RESTRICT row=frame_enc_cache->dc_dec.PlaneRow(c, y);
                              std::cout<<row[x]<<",";
                      }
              }
      }
  }
  if (aux_out != nullptr) {
    aux_out->InspectImage3F("compressed_image:InitializeFrameEncCache:dc_dec",
                            frame_enc_cache->dc_dec);
  }
}

SIMD_ATTR void InitializeEncCache(const FrameHeader &frame_header,
                                  const GroupHeader &group_header,
                                  const FrameEncCache &frame_enc_cache,
                                  const Rect &group_rect, EncCache *enc_cache) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  PIK_ASSERT(!enc_cache->initialized);

  const size_t full_xsize_blocks = frame_enc_cache.dc_dec.xsize();
  const size_t full_ysize_blocks = frame_enc_cache.dc_dec.ysize();
  const size_t x0_blocks = group_rect.x0() / N;
  const size_t y0_blocks = group_rect.y0() / N;

  enc_cache->xsize_blocks = group_rect.xsize() / N;
  enc_cache->ysize_blocks = group_rect.ysize() / N;
  enc_cache->predict_lf = false; // frame_header.predict_lf;
  enc_cache->predict_hf = false; // frame_header.predict_hf;
  enc_cache->grayscale_opt = frame_enc_cache.grayscale_opt;

  enc_cache->dc_dec =
      Image3F(enc_cache->xsize_blocks + 2, enc_cache->ysize_blocks + 2);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < enc_cache->ysize_blocks + 2; y++) {
      const size_t y_src = SourceCoord(y + y0_blocks, full_ysize_blocks);
      const float *row_src = frame_enc_cache.dc_dec.ConstPlaneRow(c, y_src);
      float *row_dc = enc_cache->dc_dec.PlaneRow(c, y);
      for (size_t x = 0; x < enc_cache->xsize_blocks + 2; x++) {
        const size_t x_src = SourceCoord(x + x0_blocks, full_xsize_blocks);
        row_dc[x] = row_src[x_src];
      }
    }
  }

  std::cout << "std_frame_dc_dec:"<<std::endl; 
  for (size_t by = 0; by < enc_cache->ysize_blocks; ++by) {
    const float *PIK_RESTRICT row_x = frame_enc_cache.dc_dec.ConstPlaneRow(0, by);
    for (size_t bx = 0; bx < enc_cache->xsize_blocks; ++bx) {
          std::cout << row_x[bx] << ",";
    }
    std::cout << std::endl;
  }

  std::cout << "std_enc_dc_dec:"<<std::endl; 
  for (size_t by = 0; by < enc_cache->ysize_blocks + 2; ++by) {
    const float *PIK_RESTRICT row_x = enc_cache->dc_dec.PlaneRow(0, by);
    for (size_t bx = 0; bx < enc_cache->xsize_blocks + 2; ++bx) {
          std::cout << row_x[bx] << ",";
    }
    std::cout << std::endl;
  }

  for (size_t by = 0; by < enc_cache->ysize_blocks; ++by) {
    const float *PIK_RESTRICT row_x = frame_enc_cache.dc_dec.ConstPlaneRow(0, by);
    const float *PIK_RESTRICT row_y = frame_enc_cache.dc_dec.ConstPlaneRow(1, by);
    const float *PIK_RESTRICT row_b = frame_enc_cache.dc_dec.ConstPlaneRow(2, by);
    /*
    for (size_t bx = 0; bx < enc_cache->xsize_blocks; ++bx) {
          std::cout << "frame_dc_dec by=" << by << " bx=" << bx
                    << " inx=" << row_x[bx]
                    << " iny=" << row_y[bx]
                    << " inb=" << row_b[bx] << std::endl;

    }
    */
  }

  const Rect coeff_rect(x0_blocks * kDCTBlockSize, y0_blocks,
                        enc_cache->xsize_blocks * kDCTBlockSize,
                        enc_cache->ysize_blocks);

  enc_cache->coeffs = CopyImage(coeff_rect, frame_enc_cache.coeffs);

  enc_cache->initialized = true;

  enc_cache->ac_strategy = frame_enc_cache.ac_strategy.Copy(Rect(
      x0_blocks, y0_blocks, enc_cache->xsize_blocks, enc_cache->ysize_blocks));
}

SIMD_ATTR void ComputeCoefficients(const Quantizer &quantizer,
                                   const ColorCorrelationMap &cmap,
                                   const Rect &cmap_rect,
                                   const FrameEncCache &frame_enc_cache,
                                   EncCache *enc_cache, PikInfo *aux_out) {
  PROFILER_FUNC;
  const size_t xsize_blocks = enc_cache->xsize_blocks;
  const size_t ysize_blocks = enc_cache->ysize_blocks;
  PIK_ASSERT(enc_cache->initialized);

  enc_cache->quant_field = CopyImage(quantizer.RawQuantField());
  ImageI &quant_field = enc_cache->quant_field;

  // TODO(user): it would be better to find & apply correlation here, when
  // quantization is chosen.

  Image3F coeffs_init;
  if (aux_out && aux_out->testing_aux.ac_prediction != nullptr) {
    coeffs_init = CopyImage(enc_cache->coeffs);
  }

  constexpr int cY = 1;
/*
  for (size_t by = 0; by < ysize_blocks; ++by) {
    const float *PIK_RESTRICT row_x = enc_cache->coeffs.ConstPlaneRow(0, by);
    const float *PIK_RESTRICT row_y = enc_cache->coeffs.ConstPlaneRow(1, by);
    const float *PIK_RESTRICT row_b = enc_cache->coeffs.ConstPlaneRow(2, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      for (int k = 0; k < 64; k++) {
        if (by == 0 && bx == 0)
          std::cout << "std_pre_llf by=" << by << " bx=" << bx << " k=" << k
                    << " inx=" << row_x[bx * 64 + k]
                    << " iny=" << row_y[bx * 64 + k]
                    << " inb=" << row_b[bx * 64 + k] << std::endl;
      }
    }
  }

  for (size_t by = 0; by < enc_cache->dc_dec.ysize(); ++by) {
    const float *PIK_RESTRICT row_x = enc_cache->dc_dec.ConstPlaneRow(0, by);
    const float *PIK_RESTRICT row_y = enc_cache->dc_dec.ConstPlaneRow(1, by);
    const float *PIK_RESTRICT row_b = enc_cache->dc_dec.ConstPlaneRow(2, by);
    for (size_t bx = 0; bx < enc_cache->dc_dec.xsize(); ++bx) {
          std::cout << "std_dc_dec by=" << by << " bx=" << bx
                    << " inx=" << row_x[bx]
                    << " iny=" << row_y[bx]
                    << " inb=" << row_b[bx] << std::endl;

    }
  }
*/
  
  Image3F pred2x2(enc_cache->dc_dec.xsize() * 2, enc_cache->dc_dec.ysize() * 2);
  
  PredictLfForEncoder(
      false, false, enc_cache->dc_dec,
      enc_cache->ac_strategy, cmap, cmap_rect, quantizer,
      frame_enc_cache.dequant_control_field, frame_enc_cache.dequant_map,
      &enc_cache->coeffs, &pred2x2);

  {
    Image3F coeffs_ac = CopyImage(enc_cache->coeffs);

    // Pre-quantized, matches what decoder will see.
    ImageF dec_ac_Y(xsize_blocks * kDCTBlockSize, ysize_blocks);
    const size_t coeffs_stride = coeffs_ac.PixelsPerRow();
    const size_t dec_ac_stride = dec_ac_Y.PixelsPerRow();

    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float *PIK_RESTRICT row_in = coeffs_ac.ConstPlaneRow(cY, by);
      float *PIK_RESTRICT row_out = dec_ac_Y.Row(by);
      AcStrategyRow ac_strategy_row = enc_cache->ac_strategy.ConstRow(by);
      size_t ty = by / kColorTileDimInBlocks;
      const uint8_t *row_quant_cf =
          cmap_rect.ConstRow(frame_enc_cache.dequant_control_field, ty);

      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        AcStrategy acs = ac_strategy_row[bx];
        if (!acs.IsFirstBlock())
          continue;
        const int32_t quant_ac = quant_field.Row(by)[bx];
        size_t tx = bx / kColorTileDimInBlocks;
        uint8_t quant_table =
            frame_enc_cache.dequant_map[row_quant_cf[tx]][quant_ac];

        quantizer.QuantizeRoundtripBlockAC(
            cY, quant_table, quant_ac, acs.GetQuantKind(),
            acs.covered_blocks_x(), acs.covered_blocks_y(),
            row_in + bx * kDCTBlockSize, coeffs_stride,
            row_out + bx * kDCTBlockSize, dec_ac_stride);
      }
    }

    for (size_t y = 0; y < ((ysize_blocks%4 == 0) ? (ysize_blocks/4):(ysize_blocks/4+1)); ++y) {
      for (size_t x = 0; x < ((xsize_blocks%4 == 0) ? (xsize_blocks/4):(xsize_blocks/4+1)); ++x) {
    for (size_t by = 0; by <4; ++by) {
	      const float *PIK_RESTRICT row_x = coeffs_ac.ConstPlaneRow(0, y*4+by);
	      const float *PIK_RESTRICT row_b = coeffs_ac.ConstPlaneRow(2, y*4+by);
      float *PIK_RESTRICT row_y = dec_ac_Y.Row(y*4+by);
      for (size_t bx = 0; bx < 4; ++bx) {
    	if((y*4+by<ysize_blocks) && (x*4+bx<xsize_blocks))
        for (int k = 0; k < 64; k++) {
            std::cout << "std_corr_in by=" << by << " bx=" << bx << " k=" << k
            << " x=" << row_x[(x*4+bx) * 64 + k]
            << " y=" << row_y[(x*4+bx) * 64 + k]
            << " b=" << row_b[(x*4+bx) * 64 + k]
			<< std::endl;
        }
      }
    }
      }
    }

    UnapplyColorCorrelationAC(cmap, cmap_rect, dec_ac_Y, &coeffs_ac);

    for (size_t y = 0; y < ((ysize_blocks%4 == 0) ? (ysize_blocks/4):(ysize_blocks/4+1)); ++y) {
          for (size_t x = 0; x < ((xsize_blocks%4 == 0) ? (xsize_blocks/4):(xsize_blocks/4+1)); ++x) {
    	  for (size_t by = 0; by < 4; ++by) {
    	  for (size_t bx = 0; bx < 4; ++bx) {
    	      const float *PIK_RESTRICT row_x = coeffs_ac.ConstPlaneRow(0, y*4+by);
    	      const float *PIK_RESTRICT row_y = coeffs_ac.ConstPlaneRow(1, y*4+by);
    	      const float *PIK_RESTRICT row_b = coeffs_ac.ConstPlaneRow(2, y*4+by);

    	if((y*4+by<ysize_blocks) && (x*4+bx<xsize_blocks))
        for (int k = 0; k < 64; k++) {
            std::cout << "std_corr_out by=" << by << " bx=" << bx << " k=" << k
                      << " x=" << row_x[(x*4+bx) * 64 + k]
                      << " y=" << row_y[(x*4+bx) * 64 + k]
                      << " b=" << row_b[(x*4+bx) * 64 + k]
                      << std::endl;
        }
    	  }
    	  }
      }
    }

    enc_cache->ac = Image3S(xsize_blocks * kDCTBlockSize, ysize_blocks);
    size_t ac_stride = enc_cache->ac.PixelsPerRow();

    for (int c = 0; c < 3; ++c) {
      for (size_t by = 0; by < ysize_blocks; ++by) {
        const float *PIK_RESTRICT row_in = coeffs_ac.PlaneRow(c, by);
        int16_t *PIK_RESTRICT row_out = enc_cache->ac.PlaneRow(c, by);
        const int32_t *row_quant = quant_field.ConstRow(by);
        AcStrategyRow ac_strategy_row = enc_cache->ac_strategy.ConstRow(by);
        size_t ty = by / kColorTileDimInBlocks;
        const uint8_t *row_quant_cf =
            cmap_rect.ConstRow(frame_enc_cache.dequant_control_field, ty);
        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          AcStrategy acs = ac_strategy_row[bx];
          if (!acs.IsFirstBlock())
            continue;
          int quant_ac = row_quant[bx];
          size_t tx = bx / kColorTileDimInBlocks;
          uint8_t quant_table =
              frame_enc_cache.dequant_map[row_quant_cf[tx]][quant_ac];
          quantizer.QuantizeBlockAC(
              quant_table, quant_ac, ac_strategy_row[bx].GetQuantKind(), c,
              acs.covered_blocks_x(), acs.covered_blocks_y(),
              row_in + bx * kDCTBlockSize, coeffs_stride,
              row_out + bx * kDCTBlockSize, ac_stride);
        }
      }
    }

    std::cout << "std_acs:" << std::endl;
    for (size_t by = 0; by < ysize_blocks; ++by) {
      AcStrategyRow ac_strategy_row = enc_cache->ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        AcStrategy acs = ac_strategy_row[bx];
        std::cout << (int)acs.RawStrategy() << ",";
      }
      std::cout << std::endl;
    }

    std::cout << "std_block:" << std::endl;
    for (size_t by = 0; by < ysize_blocks; ++by) {
      AcStrategyRow ac_strategy_row = enc_cache->ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        AcStrategy acs = ac_strategy_row[bx];
          std::cout << (int)acs.Block() <<",";
      }
      std::cout << std::endl;
    }

    std::cout << "std_qf:" << std::endl;
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const int32_t *row_quant = quant_field.ConstRow(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        int quant_ac = row_quant[bx];
        std::cout << quant_ac << ",";
      }
      std::cout << std::endl;
    }

  for(int c = 0; c< 3; c++) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      int16_t *PIK_RESTRICT row_out = enc_cache->ac.PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        std::cout <<std::dec<< "std_ac c="<<c<<" id:" << by * xsize_blocks + bx;
        for (int i = 0; i < 64; i++) {
            std::cout <<std::dec<<","<< (int)row_out[bx * kDCTBlockSize + i];
        }
        std::cout << std::endl;
      }
    }
  }
  
  }
}

PaddedBytes EncodeToBitstream(const EncCache &enc_cache, const Rect &rect,
                              const Quantizer &quantizer,
                              const NoiseParams &noise_params, bool fast_mode,
                              MultipassHandler *handler, PikInfo *info) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  PIK_ASSERT(rect.x0() % kTileDim == 0);
  PIK_ASSERT(rect.xsize() % N == 0);
  PIK_ASSERT(rect.y0() % kTileDim == 0);
  PIK_ASSERT(rect.ysize() % N == 0);
  const size_t xsize_blocks = rect.xsize() / N;
  const size_t ysize_blocks = rect.ysize() / N;
  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);
  const Rect group_acs_qf_area_rect(rect.x0() / N, rect.y0() / N, xsize_blocks,
                                    ysize_blocks);
  const Rect tile_rect(rect.x0() / kTileDim, rect.y0() / kTileDim, xsize_tiles,
                       ysize_tiles);

  PikImageSizeInfo *ac_info =
      info != nullptr ? &info->layers[kLayerAC] : nullptr;
  std::string noise_code = EncodeNoise(noise_params);

  const Rect ac_rect(N * rect.x0(), rect.y0() / N, N * rect.xsize(),
                     rect.ysize() / N);

  // TODO(veluca): do not allocate every call, allocate only what is actually
  // needed.
  Image3S ac(enc_cache.ac.xsize(), enc_cache.ac.ysize());
  size_t enc_stride = enc_cache.ac.PixelsPerRow();
  size_t ac_stride = ac.PixelsPerRow();
  // Scatter coefficients. TODO(veluca): remove when large blocks are
  // encoded all at once.
  for (size_t c = 0; c < 3; c++) {
    for (size_t by = 0; by < ysize_blocks; by++) {
      AcStrategyRow acs_row =
          enc_cache.ac_strategy.ConstRow(group_acs_qf_area_rect, by);
      const int16_t *row_in = ac_rect.ConstPlaneRow(enc_cache.ac, c, by);
      int16_t *row_out = ac_rect.PlaneRow(&ac, c, by);
      for (size_t bx = 0; bx < xsize_blocks; bx++) {
        AcStrategy acs = acs_row[bx];
        acs.ScatterCoefficients(row_in + kDCTBlockSize * bx, enc_stride,
                                row_out + kDCTBlockSize * bx, ac_stride);
      }
    }
  }

  int32_t order[kOrderContexts * kDCTBlockSize];
  ComputeCoeffOrder(ac, ac_rect, order);

  std::cout << "std_order: " << std::endl;
  for (int i = 0; i < 64; i++)
    std::cout << order[i] << ",";
  std::cout << std::endl;

  std::string order_code = EncodeCoeffOrders(order, info);

  std::vector<std::vector<Token>> ac_tokens(1);

  for (size_t y = 0; y < ysize_tiles; y++) {
    for (size_t x = 0; x < xsize_tiles; x++) {
      const Rect tile_rect(x * kTileDimInBlocks, y * kTileDimInBlocks,
                           kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                           ysize_blocks);
      TokenizeCoefficients(order, tile_rect, ac, &ac_tokens[0]);
    }
  }

  std::vector<uint8_t> context_map;
  std::vector<ANSEncodingData> codes;
  std::string histo_code = "";
  if (fast_mode) {
    histo_code =
        BuildAndEncodeHistogramsFast(ac_tokens, &codes, &context_map, ac_info);
  } else {
    histo_code = BuildAndEncodeHistograms(kNumContexts, ac_tokens, &codes,
                                          &context_map, ac_info);
  }

  std::string ac_code = WriteTokens(ac_tokens[0], codes, context_map, ac_info);

  if (info) {
    info->layers[kLayerHeader].total_size += noise_code.size();
  }

  PaddedBytes out(noise_code.size() + order_code.size() + histo_code.size() +
                  ac_code.size());
  size_t byte_pos = 0;
  Append(noise_code, &out, &byte_pos);
  Append(order_code, &out, &byte_pos);
  Append(histo_code, &out, &byte_pos);
  Append(ac_code, &out, &byte_pos);

  // TODO(veluca): fix this with DC supergroups.
  float output_size_estimate = out.size() - ac_code.size() - histo_code.size();
  std::vector<std::array<size_t, 256>> counts(kNumContexts);
  size_t extra_bits = 0;
  for (const auto &token_list : ac_tokens) {
    for (const auto &token : token_list) {
      counts[token.context][token.symbol]++;
      extra_bits += token.nbits;
    }
  }
  float entropy_coded_bits = 0;
  for (size_t ctx = 0; ctx < kNumContexts; ctx++) {
    size_t total =
        std::accumulate(counts[ctx].begin(), counts[ctx].end(), size_t(0));
    if (total == 0)
      continue; // Prevent div by zero.
    double entropy = 0;
    for (size_t i = 0; i < 256; i++) {
      double p = 1.0 * counts[ctx][i] / total;
      if (p > 1e-4) {
        entropy -= p * std::log(p);
      }
    }
    entropy_coded_bits += entropy * total / std::log(2);
  }
  output_size_estimate +=
      static_cast<float>(extra_bits + entropy_coded_bits) / kBitsPerByte;
  if (info != nullptr)
    info->entropy_estimate = output_size_estimate;
  return out;
}


PaddedBytes hls_EncodeToBitstream(const EncCache &enc_cache, const Rect &rect,
                              const Quantizer &quantizer,
                              const NoiseParams &noise_params, bool fast_mode,
                              MultipassHandler *handler, PikInfo *info,

							  ap_uint<32> *ac_x,
							  ap_uint<32> *ac_y,
							  ap_uint<32> *ac_b,
							  ap_uint<32> *k2_order) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  PIK_ASSERT(rect.x0() % kTileDim == 0);
  PIK_ASSERT(rect.xsize() % N == 0);
  PIK_ASSERT(rect.y0() % kTileDim == 0);
  PIK_ASSERT(rect.ysize() % N == 0);
  const size_t xsize_blocks = rect.xsize() / N;
  const size_t ysize_blocks = rect.ysize() / N;
  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);
  const Rect group_acs_qf_area_rect(rect.x0() / N, rect.y0() / N, xsize_blocks,
                                    ysize_blocks);
  const Rect tile_rect(rect.x0() / kTileDim, rect.y0() / kTileDim, xsize_tiles,
                       ysize_tiles);

  PikImageSizeInfo *ac_info =
      info != nullptr ? &info->layers[kLayerAC] : nullptr;
  std::string noise_code = EncodeNoise(noise_params);

  const Rect ac_rect(N * rect.x0(), rect.y0() / N, N * rect.xsize(),
                     rect.ysize() / N);

  // TODO(veluca): do not allocate every call, allocate only what is actually
  // needed.
  Image3S ac(enc_cache.ac.xsize(), enc_cache.ac.ysize());
  size_t enc_stride = enc_cache.ac.PixelsPerRow();
  size_t ac_stride = ac.PixelsPerRow();
  // Scatter coefficients. TODO(veluca): remove when large blocks are
  // encoded all at once.
  for (size_t c = 0; c < 3; c++) {
    for (size_t by = 0; by < ysize_blocks; by++) {
      AcStrategyRow acs_row =
          enc_cache.ac_strategy.ConstRow(group_acs_qf_area_rect, by);
      const int16_t *row_in = ac_rect.ConstPlaneRow(enc_cache.ac, c, by);
      int16_t *row_out = ac_rect.PlaneRow(&ac, c, by);
      for (size_t bx = 0; bx < xsize_blocks; bx++) {
        AcStrategy acs = acs_row[bx];
        acs.ScatterCoefficients(row_in + kDCTBlockSize * bx, enc_stride,
                                row_out + kDCTBlockSize * bx, ac_stride);
      }
    }
  }

  std::cout << "std_ac_x:" << std::endl;
    for (size_t by = 0; by < ysize_blocks; by++) {
      int16_t *PIK_RESTRICT row_out = ac.PlaneRow(0, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
    	std::cout<<"id:"<<by*xsize_blocks+bx<<std::endl;
        for (int i = 0; i < 64; i++)
          std::cout << row_out[bx * 64 + i] << ",";
        std::cout << std::endl;
      }
    }

    std::cout << "std_ac_y:" << std::endl;
    for (size_t by = 0; by < ysize_blocks; by++) {
      int16_t *PIK_RESTRICT row_out = ac.PlaneRow(1, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
    	std::cout<<"id:"<<by*xsize_blocks+bx<<std::endl;
        for (int i = 0; i < 64; i++)
          std::cout << row_out[bx * 64 + i] << ",";
        std::cout << std::endl;
      }
    }

    std::cout << "std_ac_b:" << std::endl;
    for (size_t by = 0; by < ysize_blocks; by++) {
      int16_t *PIK_RESTRICT row_out = ac.PlaneRow(2, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
    	std::cout<<"id:"<<by*xsize_blocks+bx<<std::endl;
        for (int i = 0; i < 64; i++)
          std::cout << row_out[bx * 64 + i] << ",";
        std::cout << std::endl;
      }
    }

#ifdef USE_K2_DATA
      for (size_t by = 0; by < ysize_blocks; ++by) {
        int16_t* PIK_RESTRICT ac_xin = ac.PlaneRow(0, by);
        int16_t* PIK_RESTRICT ac_yin = ac.PlaneRow(1, by);
        int16_t* PIK_RESTRICT ac_bin = ac.PlaneRow(2, by);

        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        	for(int i=0;i<8;i++){
        		for(int j=0;j<8;j++){
        			ac_xin[bx*kDCTBlockSize+8*i+j]=ac_x[(by*xsize_blocks+bx)*64+8*i+j];
        		    ac_yin[bx*kDCTBlockSize+8*i+j]=ac_y[(by*xsize_blocks+bx)*64+8*i+j];
        		    ac_bin[bx*kDCTBlockSize+8*i+j]=ac_b[(by*xsize_blocks+bx)*64+8*i+j];
        	  }
           }
        }
      }
#endif

  int32_t order[kOrderContexts * kDCTBlockSize];
  ComputeCoeffOrder(ac, ac_rect, order);

  std::cout << "std_order: " << std::endl;
  for (int i = 0; i < 64*3; i++)
    std::cout << order[i] << ",";
  std::cout << std::endl;

#ifdef USE_K2_DATA
      for (size_t i = 0; i < 64*3; ++i) {
    	  order[i]=k2_order[i];
      }
#endif

  std::string order_code = EncodeCoeffOrders(order, info);

  std::vector<std::vector<Token>> ac_tokens(1);

  for (size_t y = 0; y < ysize_tiles; y++) {
    for (size_t x = 0; x < xsize_tiles; x++) {
      const Rect tile_rect(x * kTileDimInBlocks, y * kTileDimInBlocks,
                           kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                           ysize_blocks);
      TokenizeCoefficients(order, tile_rect, ac, &ac_tokens[0]);
    }
  }

  std::vector<uint8_t> context_map;
  std::vector<ANSEncodingData> codes;
  std::string histo_code = "";
  if (fast_mode) {
    histo_code =
        BuildAndEncodeHistogramsFast(ac_tokens, &codes, &context_map, ac_info);
  } else {
    histo_code = BuildAndEncodeHistograms(kNumContexts, ac_tokens, &codes,
                                          &context_map, ac_info);
  }

  std::string ac_code = WriteTokens(ac_tokens[0], codes, context_map, ac_info);

  if (info) {
    info->layers[kLayerHeader].total_size += noise_code.size();
  }

  PaddedBytes out(noise_code.size() + order_code.size() + histo_code.size() +
                  ac_code.size());
  size_t byte_pos = 0;
  Append(noise_code, &out, &byte_pos);
  Append(order_code, &out, &byte_pos);
  Append(histo_code, &out, &byte_pos);
  Append(ac_code, &out, &byte_pos);

  // TODO(veluca): fix this with DC supergroups.
  float output_size_estimate = out.size() - ac_code.size() - histo_code.size();
  std::vector<std::array<size_t, 256>> counts(kNumContexts);
  size_t extra_bits = 0;
  for (const auto &token_list : ac_tokens) {
    for (const auto &token : token_list) {
      counts[token.context][token.symbol]++;
      extra_bits += token.nbits;
    }
  }
  float entropy_coded_bits = 0;
  for (size_t ctx = 0; ctx < kNumContexts; ctx++) {
    size_t total =
        std::accumulate(counts[ctx].begin(), counts[ctx].end(), size_t(0));
    if (total == 0)
      continue; // Prevent div by zero.
    double entropy = 0;
    for (size_t i = 0; i < 256; i++) {
      double p = 1.0 * counts[ctx][i] / total;
      if (p > 1e-4) {
        entropy -= p * std::log(p);
      }
    }
    entropy_coded_bits += entropy * total / std::log(2);
  }
  output_size_estimate +=
      static_cast<float>(extra_bits + entropy_coded_bits) / kBitsPerByte;
  if (info != nullptr)
    info->entropy_estimate = output_size_estimate;
  return out;
}

template <bool first> class Dequant {
public:
  Dequant(const Quantizer &quantizer) : quantizer_(quantizer) {
    dequant_matrices_ = quantizer.DequantMatrix(0, kQuantKindDCT8, 0);
    inv_global_scale_ = quantizer.InvGlobalScale();
  }

  // Dequantizes and inverse color-transforms one tile, i.e. the window
  // `rect` (in block units) within the output image `group_dec_cache->ac`.
  // Reads the rect `rect16` (in block units) in `img_ac16`. Reads and write
  // only to the `block_group_rect` part of ac_strategy/quant_field.
  SIMD_ATTR void DoAC(const Rect &rect16, const Image3S &img_ac16,
                      const Rect &rect, const Rect &block_group_rect,
                      const ImageI &img_ytox, const ImageI &img_ytob,
                      const Rect &cmap_rect,
                      FrameDecCache *PIK_RESTRICT frame_dec_cache,
                      GroupDecCache *PIK_RESTRICT group_dec_cache,
                      PikInfo *aux_out) const {
    PROFILER_FUNC;
    PIK_ASSERT(SameSize(rect, rect16));
    const size_t xsize = rect.xsize(); // [blocks]
    const size_t ysize = rect.ysize();
    PIK_ASSERT(img_ac16.xsize() % kDCTBlockSize == 0);
    PIK_ASSERT(xsize <= img_ac16.xsize() / kDCTBlockSize);
    PIK_ASSERT(ysize <= img_ac16.ysize());
    PIK_ASSERT(SameSize(img_ytox, img_ytob));

    using D = SIMD_FULL(float);
    constexpr D d;
    constexpr SIMD_PART(int16_t, D::N) d16;
    constexpr SIMD_PART(int32_t, D::N) d32;

    // Rect representing the current tile inside the current group, in an image
    // in which each block is 1x1.
    const Rect block_tile_group_rect(block_group_rect.x0() + rect.x0(),
                                     block_group_rect.y0() + rect.y0(),
                                     rect.xsize(), rect.ysize());

    const size_t x0_cmap = rect.x0() / kColorTileDimInBlocks;
    const size_t y0_cmap = rect.y0() / kColorTileDimInBlocks;
    const size_t x0_dct = rect.x0() * kDCTBlockSize;
    const size_t x0_dct16 = rect16.x0() * kDCTBlockSize;

    // TODO(veluca): get rid of acs.Block() and only use acs.IsFirst()
    for (size_t by = 0; by < ysize; ++by) {
      const size_t ty = by / kColorTileDimInBlocks;
      const int16_t *PIK_RESTRICT row_16[3] = {
          img_ac16.PlaneRow(0, by + rect16.y0()) + x0_dct16,
          img_ac16.PlaneRow(1, by + rect16.y0()) + x0_dct16,
          img_ac16.PlaneRow(2, by + rect16.y0()) + x0_dct16};
      const int *PIK_RESTRICT row_quant_field =
          block_tile_group_rect.ConstRow(frame_dec_cache->raw_quant_field, by);
      static_assert(kColorTileDimInBlocks == kTileDimInBlocks,
                    "Quantization table selection assumes that color tile and "
                    "tiles have the same size!");
      const uint8_t *PIK_RESTRICT row_quant_cf =
          cmap_rect.ConstRow(frame_dec_cache->dequant_control_field,
                             ty + y0_cmap) +
          x0_cmap;
      const int *PIK_RESTRICT row_cmap[3] = {
          cmap_rect.ConstRow(img_ytox, ty + y0_cmap) + x0_cmap, nullptr,
          cmap_rect.ConstRow(img_ytob, ty + y0_cmap) + x0_cmap,
      };
      float *PIK_RESTRICT row[3] = {
          group_dec_cache->ac.PlaneRow(0, rect.y0() + by) + x0_dct,
          group_dec_cache->ac.PlaneRow(1, rect.y0() + by) + x0_dct,
          group_dec_cache->ac.PlaneRow(2, rect.y0() + by) + x0_dct,
      };

      AcStrategyRow ac_strategy_row =
          frame_dec_cache->ac_strategy.ConstRow(block_tile_group_rect, by);
      for (size_t bx = 0; bx < xsize; ++bx) {
        const auto scaled_dequant =
            set1(d, SafeDiv(inv_global_scale_, row_quant_field[bx]));
        const size_t tx = bx / kColorTileDimInBlocks;
        uint8_t quant_table =
            frame_dec_cache
                ->dequant_map[row_quant_cf[tx]][row_quant_field[bx] - 1];

        size_t kind = ac_strategy_row[bx].GetQuantKind();
        const float *PIK_RESTRICT dequant_matrix[3] = {
            &dequant_matrices_[quantizer_.DequantMatrixOffset(quant_table, kind,
                                                              0) +
                               ac_strategy_row[bx].Block() * kDCTBlockSize],
            &dequant_matrices_[quantizer_.DequantMatrixOffset(quant_table, kind,
                                                              1) +
                               ac_strategy_row[bx].Block() * kDCTBlockSize],
            &dequant_matrices_[quantizer_.DequantMatrixOffset(quant_table, kind,
                                                              2) +
                               ac_strategy_row[bx].Block() * kDCTBlockSize],
        };
        const auto x_cc_mul =
            set1(d, ColorCorrelationMap::YtoX(1.0f, row_cmap[0][tx]));
        const auto b_cc_mul =
            set1(d, ColorCorrelationMap::YtoB(1.0f, row_cmap[2][tx]));
        for (size_t k = 0; k < kDCTBlockSize; k += d.N) {
          const size_t x = bx * kDCTBlockSize + k;

          const auto x_mul = load(d, dequant_matrix[0] + k) * scaled_dequant;
          const auto y_mul = load(d, dequant_matrix[1] + k) * scaled_dequant;
          const auto b_mul = load(d, dequant_matrix[2] + k) * scaled_dequant;

          const auto quantized_x16 = load(d16, row_16[0] + x);
          const auto quantized_y16 = load(d16, row_16[1] + x);
          const auto quantized_b16 = load(d16, row_16[2] + x);
          const auto quantized_x =
              convert_to(d, convert_to(d32, quantized_x16));
          const auto quantized_y =
              convert_to(d, convert_to(d32, quantized_y16));
          const auto quantized_b =
              convert_to(d, convert_to(d32, quantized_b16));

          const auto dequant_x_cc = AdjustQuantBias<0>(quantized_x) * x_mul;
          const auto dequant_y = AdjustQuantBias<1>(quantized_y) * y_mul;
          const auto dequant_b_cc = AdjustQuantBias<2>(quantized_b) * b_mul;

          const auto dequant_x = mul_add(x_cc_mul, dequant_y, dequant_x_cc);
          const auto dequant_b = mul_add(b_cc_mul, dequant_y, dequant_b_cc);

          if (first) {
            store(dequant_x, d, row[0] + x);
            store(dequant_y, d, row[1] + x);
            store(dequant_b, d, row[2] + x);
          } else {
            store(dequant_x + load(d, row[0] + x), d, row[0] + x);
            store(dequant_y + load(d, row[1] + x), d, row[1] + x);
            store(dequant_b + load(d, row[2] + x), d, row[2] + x);
          }
        }
      }
    }
  }

private:
  static PIK_INLINE float SafeDiv(float num, int32_t div) {
    return div == 0 ? 1E10f : num / div;
  }

  // AC dequant
  const float *PIK_RESTRICT dequant_matrices_;
  float inv_global_scale_;
  const Quantizer &quantizer_;
};

template <bool first>
bool DecodeFromBitstream(const FrameHeader &frame_header,
                         const GroupHeader &header,
                         const PaddedBytes &compressed, BitReader *reader,
                         const Rect &group_rect, MultipassHandler *handler,
                         const size_t xsize_blocks, const size_t ysize_blocks,
                         const ColorCorrelationMap &cmap, const Rect &cmap_rect,
                         NoiseParams *noise_params, const Quantizer &quantizer,
                         FrameDecCache *PIK_RESTRICT frame_dec_cache,
                         GroupDecCache *PIK_RESTRICT group_dec_cache,
                         PikInfo *aux_out) {
  PROFILER_FUNC;

  PIK_RETURN_IF_ERROR(DecodeNoise(reader, noise_params));

  PIK_ASSERT(group_rect.x0() % kBlockDim == 0);
  PIK_ASSERT(group_rect.y0() % kBlockDim == 0);
  const size_t x0_blocks = DivCeil(group_rect.x0(), kBlockDim);
  const size_t y0_blocks = DivCeil(group_rect.y0(), kBlockDim);
  const Rect group_acs_qf_rect(x0_blocks, y0_blocks, xsize_blocks,
                               ysize_blocks);

  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);
  const size_t num_tiles = xsize_tiles * ysize_tiles;

  group_dec_cache->InitOnce(xsize_blocks, ysize_blocks);

  int coeff_order[kOrderContexts * kDCTBlockSize];
  for (size_t c = 0; c < kOrderContexts; ++c) {
    DecodeCoeffOrder(&coeff_order[c * kDCTBlockSize], reader);
  }
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  ANSCode code;
  std::vector<uint8_t> context_map;
  // Histogram data size is small and does not require parallelization.
  PIK_RETURN_IF_ERROR(
      DecodeHistograms(reader, kNumContexts, 256, &code, &context_map));
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  Dequant<first> dequant(quantizer);

  SIMD_ALIGN int16_t unscattered[AcStrategy::kMaxCoeffArea];
  const size_t stride = group_dec_cache->quantized_ac.PixelsPerRow();

  ANSSymbolReader ac_decoder(&code);
  for (size_t task = 0; task < num_tiles; ++task) {
    const size_t tile_x = task % xsize_tiles;
    const size_t tile_y = task / xsize_tiles;
    const Rect rect(tile_x * kTileDimInBlocks, tile_y * kTileDimInBlocks,
                    kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                    ysize_blocks);
    const Rect quantized_rect(0, 0, rect.xsize(), rect.ysize());

    if (!DecodeAC(context_map, coeff_order, reader, &ac_decoder,
                  &group_dec_cache->quantized_ac, rect,
                  &group_dec_cache->num_nzeroes)) {
      return PIK_FAILURE("Failed to decode AC.");
    }
    // Unscatter coefficients. TODO(veluca): remove when large blocks are
    // encoded all at once.
    for (size_t c = 0; c < 3; c++) {
      for (size_t by = 0; by < rect.ysize(); by++) {
        AcStrategyRow acs_row = frame_dec_cache->ac_strategy.ConstRow(
            group_acs_qf_rect, by + rect.y0());
        int16_t *PIK_RESTRICT row =
            group_dec_cache->quantized_ac.PlaneRow(c, by);
        for (size_t bx = 0; bx < rect.xsize(); bx++) {
          AcStrategy acs = acs_row[bx + rect.x0()];
          if (!acs.IsFirstBlock())
            continue;
          int16_t *block = row + bx * kDCTBlockSize;
          acs.GatherCoefficients(block, stride, unscattered,
                                 acs.covered_blocks_x() * kDCTBlockSize);
          for (size_t i = 0; i < acs.covered_blocks_y(); i++) {
            memcpy(block + stride * i,
                   unscattered + acs.covered_blocks_x() * kDCTBlockSize * i,
                   sizeof(int16_t) * acs.covered_blocks_x() * kDCTBlockSize);
          }
        }
      }
    }

    dequant.DoAC(quantized_rect, group_dec_cache->quantized_ac, rect,
                 group_acs_qf_rect, cmap.ytox_map, cmap.ytob_map, cmap_rect,
                 frame_dec_cache, group_dec_cache, aux_out);
  }
  if (!ac_decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  return true;
}

template bool DecodeFromBitstream<true>(
    const FrameHeader &, const GroupHeader &, const PaddedBytes &, BitReader *,
    const Rect &, MultipassHandler *, const size_t, const size_t,
    const ColorCorrelationMap &, const Rect &, NoiseParams *, const Quantizer &,
    FrameDecCache *PIK_RESTRICT, GroupDecCache *PIK_RESTRICT, PikInfo *);

template bool DecodeFromBitstream<false>(
    const FrameHeader &, const GroupHeader &, const PaddedBytes &, BitReader *,
    const Rect &, MultipassHandler *, const size_t, const size_t,
    const ColorCorrelationMap &, const Rect &, NoiseParams *, const Quantizer &,
    FrameDecCache *PIK_RESTRICT, GroupDecCache *PIK_RESTRICT, PikInfo *);

void DequantImageAC(const Quantizer &quantizer, const ColorCorrelationMap &cmap,
                    const Rect &cmap_rect, const Image3S &quantized_ac,
                    FrameDecCache *frame_dec_cache,
                    GroupDecCache *group_dec_cache, const Rect &group_rect,
                    PikInfo *aux_out) {
  PROFILER_ZONE("dequant");

  // Caller must have allocated/filled quantized_dc/ac.
  PIK_CHECK(quantized_ac.xsize() ==
                quantizer.RawQuantField().xsize() * kDCTBlockSize &&
            quantized_ac.ysize() == quantizer.RawQuantField().ysize());

  const size_t xsize_blocks = quantizer.RawQuantField().xsize();
  const size_t ysize_blocks = quantizer.RawQuantField().ysize();
  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);

  // Only one pass for roundtrips.
  Dequant</*first=*/true> dequant(quantizer);

  PIK_ASSERT(group_rect.x0() % kBlockDim == 0 &&
             group_rect.y0() % kBlockDim == 0 &&
             group_rect.xsize() % kBlockDim == 0 &&
             group_rect.ysize() % kBlockDim == 0);

  const Rect block_group_rect(
      group_rect.x0() / kBlockDim, group_rect.y0() / kBlockDim,
      group_rect.xsize() / kBlockDim, group_rect.ysize() / kBlockDim);

  for (size_t idx_tile = 0; idx_tile < xsize_tiles * ysize_tiles; ++idx_tile) {
    const size_t tile_x = idx_tile % xsize_tiles;
    const size_t tile_y = idx_tile / xsize_tiles;
    const Rect rect(tile_x * kTileDimInBlocks, tile_y * kTileDimInBlocks,
                    kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                    ysize_blocks);

    dequant.DoAC(rect, quantized_ac, rect, block_group_rect, cmap.ytox_map,
                 cmap.ytob_map, cmap_rect, frame_dec_cache, group_dec_cache,
                 aux_out);
  }
}

static SIMD_ATTR void
InverseIntegralTransform(const size_t xsize_blocks, const size_t ysize_blocks,
                         const Image3F &ac_image,
                         const AcStrategyImage &ac_strategy,
                         const Rect &acs_rect, Image3F *PIK_RESTRICT idct,
                         const Rect &idct_rect, size_t downsample) {
  PROFILER_ZONE("IDCT");

  constexpr size_t N = kBlockDim;
  const size_t idct_stride = idct->PixelsPerRow();
  const size_t ac_per_row = ac_image.PixelsPerRow();

  if (downsample == 1) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const AcStrategyRow &acs_row = ac_strategy.ConstRow(acs_rect, by);
      for (int c = 0; c < 3; ++c) {
        const float *PIK_RESTRICT ac_row = ac_image.ConstPlaneRow(c, by);
        float *PIK_RESTRICT idct_row = idct_rect.PlaneRow(idct, c, by * N);

        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const float *PIK_RESTRICT ac_pos = ac_row + bx * kDCTBlockSize;
          const AcStrategy &acs = acs_row[bx];
          float *PIK_RESTRICT idct_pos = idct_row + bx * N;

          acs.TransformToPixels(ac_pos, ac_per_row, idct_pos, idct_stride);
        }
      }
    }
  } else {
    float mean_mul = 1.0f / (downsample * downsample);
    PIK_ASSERT(downsample == 2 || downsample == 4 || downsample == 8);
    size_t N_downsample = N / downsample;
    SIMD_ALIGN float pixels[AcStrategy::kMaxCoeffArea];

    for (size_t by = 0; by < ysize_blocks; ++by) {
      const AcStrategyRow &acs_row = ac_strategy.ConstRow(acs_rect, by);
      for (int c = 0; c < 3; ++c) {
        const float *PIK_RESTRICT ac_row = ac_image.ConstPlaneRow(c, by);
        float *PIK_RESTRICT idct_row =
            idct_rect.PlaneRow(idct, c, by * N_downsample);

        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const float *PIK_RESTRICT ac_pos = ac_row + bx * kDCTBlockSize;
          const AcStrategy &acs = acs_row[bx];
          float *PIK_RESTRICT idct_pos = idct_row + bx * N_downsample;
          if (!acs.IsFirstBlock())
            continue;

          acs.TransformToPixels(ac_pos, ac_per_row, pixels,
                                acs.covered_blocks_x() * N);
          for (size_t y = 0; y < acs.covered_blocks_y() * N_downsample; y++) {
            for (size_t x = 0; x < acs.covered_blocks_x() * N_downsample; x++) {
              float sum = 0.0f;
              for (size_t iy = 0; iy < downsample; iy++) {
                for (size_t ix = 0; ix < downsample; ix++) {
                  sum += pixels[(y * downsample + iy) * N *
                                    acs.covered_blocks_x() +
                                x * downsample + ix];
                }
              }
              idct_pos[y * idct_stride + x] = sum * mean_mul;
            }
          }
        }
      }
    }
  }
}

void ReconOpsinImage(const FrameHeader &frame_header, const GroupHeader &header,
                     const Quantizer &quantizer, const Rect &block_group_rect,
                     FrameDecCache *PIK_RESTRICT frame_dec_cache,
                     GroupDecCache *PIK_RESTRICT group_dec_cache,
                     Image3F *PIK_RESTRICT idct, const Rect &idct_rect,
                     PikInfo *aux_out, size_t downsample) {
  PROFILER_ZONE("ReconOpsinImage");
  constexpr size_t N = kBlockDim;
  const size_t xsize_blocks = block_group_rect.xsize();
  const size_t ysize_blocks = block_group_rect.ysize();
  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);
  const bool predict_lf = frame_header.predict_lf;
  const bool predict_hf = frame_header.predict_hf;

  // TODO(veluca): this should probably happen upon dequantization of DC. Also,
  // we should consider doing something similar for AC.
  if (frame_header.flags & FrameHeader::kGrayscaleOpt) {
    PROFILER_ZONE("GrayscaleRestoreXB");
    GetGrayXyb()->RestoreXB(&group_dec_cache->dc);
  }

  if (aux_out && aux_out->testing_aux.ac_prediction != nullptr) {
    PROFILER_ZONE("Copy ac_prediction");
    *aux_out->testing_aux.ac_prediction = CopyImage(group_dec_cache->ac);
  }

  // Sets dcoeffs.0 from DC (for DCT blocks) and updates HVD.
  Image3F *PIK_RESTRICT ac64 = &group_dec_cache->ac;

  // Currently llf is temporary storage, but it will be more persistent
  // in tile-wise processing.
  ComputeLlf(group_dec_cache->dc, frame_dec_cache->ac_strategy,
             block_group_rect, &group_dec_cache->llf);

  if (predict_lf) {
    // dc2x2 plane is borrowed for temporary storage.
    PredictLf(frame_dec_cache->ac_strategy, block_group_rect,
              group_dec_cache->llf,
              const_cast<ImageF *>(&group_dec_cache->pred2x2.Plane(0)),
              &group_dec_cache->lf2x2);
  }
  ZeroFillImage(&group_dec_cache->pred2x2);

  // Compute the border of pred2x2.
  if (predict_hf) {
    PROFILER_ZONE("Predict HF");
    AcStrategy acs(AcStrategy::Type::DCT, 0);
    const size_t pred2x2_stride = group_dec_cache->pred2x2.PixelsPerRow();
    if (predict_lf) {
      const size_t lf2x2_stride = group_dec_cache->lf2x2.PixelsPerRow();
      float block[N * N] = {};
      for (size_t c = 0; c < 3; c++) {
        for (size_t x : {0UL, xsize_blocks + 1}) {
          for (size_t y = 0; y < ysize_blocks + 2; y++) {
            const float *row_llf = group_dec_cache->llf.ConstPlaneRow(c, y);
            const float *row_lf2x2 =
                group_dec_cache->lf2x2.ConstPlaneRow(c, 2 * y);
            float *row_pred2x2 = group_dec_cache->pred2x2.PlaneRow(c, 2 * y);
            block[0] = row_llf[x];
            block[1] = row_lf2x2[2 * x + 1];
            block[N] = row_lf2x2[lf2x2_stride + 2 * x];
            block[N + 1] = row_lf2x2[lf2x2_stride + 2 * x + 1];
            acs.DC2x2FromLowFrequencies(block, 0 /*not used*/,
                                        row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
        for (size_t y : {0UL, ysize_blocks + 1}) {
          const float *row_llf = group_dec_cache->llf.ConstPlaneRow(c, y);
          const float *row_lf2x2 =
              group_dec_cache->lf2x2.ConstPlaneRow(c, 2 * y);
          float *row_pred2x2 = group_dec_cache->pred2x2.PlaneRow(c, 2 * y);
          for (size_t x = 0; x < xsize_blocks + 2; x++) {
            block[0] = row_llf[x];
            block[1] = row_lf2x2[2 * x + 1];
            block[N] = row_lf2x2[lf2x2_stride + 2 * x];
            block[N + 1] = row_lf2x2[lf2x2_stride + 2 * x + 1];
            acs.DC2x2FromLowFrequencies(block, 0 /*not used*/,
                                        row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
      }
    } else {
      const size_t llf_stride = group_dec_cache->llf.PixelsPerRow();
      for (size_t c = 0; c < 3; c++) {
        for (size_t x : {0UL, xsize_blocks + 1}) {
          for (size_t y = 0; y < ysize_blocks + 2; y++) {
            const float *row_llf = group_dec_cache->llf.ConstPlaneRow(c, y);
            float *row_pred2x2 = group_dec_cache->pred2x2.PlaneRow(c, 2 * y);
            acs.DC2x2FromLowestFrequencies(row_llf + x, llf_stride,
                                           row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
        for (size_t y : {0UL, ysize_blocks + 1}) {
          const float *row_llf = group_dec_cache->llf.ConstPlaneRow(c, y);
          float *row_pred2x2 = group_dec_cache->pred2x2.PlaneRow(c, 2 * y);
          for (size_t x = 0; x < xsize_blocks + 2; x++) {
            acs.DC2x2FromLowestFrequencies(row_llf + x, llf_stride,
                                           row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
      }
    }
  }

  // tile_stage is used to make calculation dispatching simple; each pixel
  // corresponds to tile. Each bit corresponds to stage:
  // * 0-th bit for calculation or lf2x2 / pred2x2 & initial LF AC update;

  Image3F *PIK_RESTRICT pred2x2_or_null =
      predict_hf ? &group_dec_cache->pred2x2 : nullptr;
  Image3F *PIK_RESTRICT lf2x2_or_null =
      predict_lf ? &group_dec_cache->lf2x2 : nullptr;

  for (size_t c = 0; c < group_dec_cache->ac.kNumPlanes; c++) {
    PROFILER_ZONE("Reset tile stages");
    // Reset tile stages.
    for (size_t ty = 0; ty < ysize_tiles; ++ty) {
      uint8_t *PIK_RESTRICT tile_stage_row =
          group_dec_cache->tile_stage.Row(ty);
      memset(tile_stage_row, 0, xsize_tiles * sizeof(uint8_t));
      tile_stage_row[xsize_tiles] = 255;
    }
    uint8_t *PIK_RESTRICT tile_stage_row =
        group_dec_cache->tile_stage.Row(ysize_tiles);
    memset(tile_stage_row, 255, (xsize_tiles + 1) * sizeof(uint8_t));

    for (size_t ty = 0; ty < ysize_tiles; ++ty) {
      for (size_t tx = 0; tx < xsize_tiles; ++tx) {
        for (size_t lfty = ty; lfty < ty + 2; ++lfty) {
          uint8_t *tile_stage_row = group_dec_cache->tile_stage.Row(lfty);
          for (size_t lftx = tx; lftx < tx + 2; ++lftx) {
            if ((tile_stage_row[lftx] & 1) != 0)
              continue;
            const Rect tile(lftx * kTileDimInBlocks, lfty * kTileDimInBlocks,
                            kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                            ysize_blocks);
            UpdateLfForDecoder(tile, predict_lf, predict_hf,
                               frame_dec_cache->ac_strategy, block_group_rect,
                               group_dec_cache->llf, ac64, pred2x2_or_null,
                               lf2x2_or_null, c);
            tile_stage_row[lftx] |= 1;
          }
        }
        if (predict_hf) {
          // TODO(user): invoke AddPredictions for (tx, ty) tile here.
        }
      }
    }
  }

  if (predict_hf) {
    // TODO(user): make UpSample4x4BlurDCT tile-wise-able.
    AddPredictions(group_dec_cache->pred2x2, frame_dec_cache->ac_strategy,
                   block_group_rect, &group_dec_cache->blur_x,
                   &group_dec_cache->ac);
  }

  PIK_ASSERT(idct_rect.xsize() == DivCeil(xsize_blocks * N, downsample));
  PIK_ASSERT(idct_rect.ysize() == DivCeil(ysize_blocks * N, downsample));
  InverseIntegralTransform(xsize_blocks, ysize_blocks, group_dec_cache->ac,
                           frame_dec_cache->ac_strategy, block_group_rect, idct,
                           idct_rect, downsample);

  if (aux_out && aux_out->testing_aux.ac_prediction != nullptr) {
    PROFILER_ZONE("Subtract ac_prediction");
    Subtract(group_dec_cache->ac, *aux_out->testing_aux.ac_prediction,
             aux_out->testing_aux.ac_prediction);
    ZeroDcValues(aux_out->testing_aux.ac_prediction,
                 frame_dec_cache->ac_strategy);
  }
}

namespace {

Status DoAdaptiveReconstruction(const Image3F &idct,
                                const FrameHeader &frame_header,
                                const Quantizer &quantizer,
                                FrameDecCache *frame_dec_cache,
                                PikInfo *aux_out, Image3F *PIK_RESTRICT out) {
  // Since no adaptive reconstruction would want us to return the `idct`
  // parameter as `out`, which would lead to either a copy or a new memory
  // handling strategy, we disallow it and require callers to avoid it.
  PIK_CHECK(frame_header.have_adaptive_reconstruction);

  AdaptiveReconstructionAux *ar_aux =
      aux_out ? &aux_out->adaptive_reconstruction_aux : nullptr;

  const Image3F *smoothed_ptr;
  Image3F smoothed;
  // If no gaborish, the smoothed and non-smoothed inputs are the same.
  if (frame_header.gaborish == GaborishStrength::kOff) {
    smoothed_ptr = &idct;
  } else {
    PIK_RETURN_IF_ERROR(
        ConvolveGaborish(idct, frame_header.gaborish, nullptr, &smoothed));
    smoothed_ptr = &smoothed;
  }

  *out = AdaptiveReconstruction(
      *smoothed_ptr, idct, quantizer, frame_dec_cache->raw_quant_field,
      frame_dec_cache->dequant_control_field, frame_dec_cache->dequant_map,
      frame_dec_cache->ar_sigma_lut_ids, frame_dec_cache->ac_strategy,
      frame_header.epf_params, ar_aux);
  return true;
}

} // namespace

Status FinalizeFrameDecoding(Image3F *PIK_RESTRICT idct, size_t xsize,
                             size_t ysize, const FrameHeader &frame_header,
                             const NoiseParams &noise_params,
                             const Quantizer &quantizer,
                             const BlockDictionary &dictionary,
                             FrameDecCache *frame_dec_cache, PikInfo *aux_out,
                             size_t downsample) {
  if (downsample == 1 && frame_header.have_adaptive_reconstruction) {
    Image3F reconstructed;
    PIK_RETURN_IF_ERROR(DoAdaptiveReconstruction(*idct, frame_header, quantizer,
                                                 frame_dec_cache, aux_out,
                                                 &reconstructed));
    *idct = std::move(reconstructed);
  }

  ApplyForwardBilinear(idct, downsample);

  dictionary.AddTo(idct, downsample);

  if (downsample == 1) {
    if (frame_header.gaborish != GaborishStrength::kOff) {
      Image3F gaborished;
      PIK_RETURN_IF_ERROR(
          ConvolveGaborish(*idct, frame_header.gaborish, nullptr, &gaborished));
      *idct = std::move(gaborished);
    }

    if (frame_header.flags & FrameHeader::kNoise) {
      PROFILER_ZONE("AddNoise");
      AddNoise(noise_params, idct);
    }
  }

  idct->ShrinkTo(DivCeil(xsize, downsample), DivCeil(ysize, downsample));
  OpsinToLinear(idct);

  return true;
}

} // namespace pik
