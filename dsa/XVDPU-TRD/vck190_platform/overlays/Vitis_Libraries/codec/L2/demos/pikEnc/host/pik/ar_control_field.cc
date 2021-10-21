// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ar_control_field.h"
#include "pik/adaptive_quantization.h"
#include "pik/adaptive_reconstruction.h"
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/opsin_inverse.h"

namespace pik {

// TODO(veluca): Remove and enable by default.
constexpr bool kUseARField = false;

void FindBestArControlField(float distance, float intensity_target,
                            const Image3F& opsin,
                            const AcStrategyImage& ac_strategy,
                            const ImageF& quant_field,
                            const DequantMatrices* dequant,
                            GaborishStrength gaborish, ThreadPool* pool,
                            ImageB* sigma_lut_ids) {
  constexpr size_t N = kBlockDim;
  size_t xsize_blocks = DivCeil(opsin.xsize(), N);
  size_t ysize_blocks = DivCeil(opsin.ysize(), N);

  *sigma_lut_ids = ImageB(xsize_blocks, ysize_blocks);
  ZeroFillImage(sigma_lut_ids);

  if (!kUseARField) return;

  float quant_dc = InitialQuantDC(distance, intensity_target);
  Quantizer quantizer(dequant, xsize_blocks, ysize_blocks);
  quantizer.SetQuantField(quant_dc, QuantField(quant_field));

  const Image3F* smoothed_ptr;
  Image3F smoothed;
  if (gaborish == GaborishStrength::kOff) {
    smoothed_ptr = &opsin;
  } else {
    PIK_CHECK(ConvolveGaborish(opsin, gaborish, pool, &smoothed));
    smoothed_ptr = &smoothed;
  }

  Image3F filt =
      DoDenoise(*smoothed_ptr, opsin, quantizer, quantizer.RawQuantField(),
                *sigma_lut_ids, ac_strategy, EpfParams());

  constexpr float kChannelWeights[3] = {1.0, 1.0, 0.3};
  const float kInvPow =
      1.0f / (kChannelWeights[0] + kChannelWeights[1] + kChannelWeights[2]);
  constexpr float kStdDevRatioThreshold = 0.75f;

  PIK_ASSERT(filt.PixelsPerRow() == opsin.PixelsPerRow());
  size_t opsin_stride = opsin.PixelsPerRow();
  size_t sigma_stride = sigma_lut_ids->PixelsPerRow();

  for (size_t by = 0; by < ysize_blocks; by++) {
    const float* PIK_RESTRICT filt_row[3] = {filt.ConstPlaneRow(0, by * N),
                                             filt.ConstPlaneRow(1, by * N),
                                             filt.ConstPlaneRow(2, by * N)};
    const float* PIK_RESTRICT in_row[3] = {
        opsin.ConstPlaneRow(0, by * N),
        opsin.ConstPlaneRow(1, by * N),
        opsin.ConstPlaneRow(2, by * N),
    };

    AcStrategyRow acs_row = ac_strategy.ConstRow(by);
    uint8_t* PIK_RESTRICT out_row = sigma_lut_ids->Row(by);
    for (size_t bx = 0; bx < xsize_blocks; bx++) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      float avg_ratio = 1;
      uint8_t lut = 0;
      for (size_t c = 0; c < 3; c++) {
        Stats stats_in;
        Stats stats_filt;
        for (size_t iy = 0; iy < acs.covered_blocks_y() * N; iy++) {
          for (size_t ix = 0; ix < acs.covered_blocks_x() * N; ix++) {
            stats_in.Notify(in_row[c][bx * N + iy * opsin_stride + ix]);
            stats_filt.Notify(filt_row[c][bx * N + iy * opsin_stride + ix]);
          }
        }
        float in_dev = stats_in.StandardDeviation();
        float filt_dev = stats_filt.StandardDeviation();

        float r = pow(filt_dev / in_dev, kChannelWeights[c]);
        if (r > 3) r = 3;
        if (r < 1e-2) r = 1e-2;
        avg_ratio *= r;
      }
      float ratio = std::pow(avg_ratio, kInvPow);
      if (ratio < kStdDevRatioThreshold) {
        lut = 1;
      }
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          out_row[bx + sigma_stride * iy + ix] = lut;
        }
      }
    }
  }
}

}  // namespace pik
