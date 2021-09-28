// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/compressed_image.h"

#include <array>

#include "gtest/gtest.h"
#include "pik/adaptive_reconstruction.h"
#include "pik/butteraugli_distance.h"
#include "pik/codec.h"
#include "pik/common.h"
#include "pik/compressed_dc.h"
#include "pik/entropy_coder.h"
#include "pik/gaborish.h"
#include "pik/gradient_map.h"
#include "pik/image_ops.h"
#include "pik/image_test_utils.h"
#include "pik/opsin_image.h"
#include "pik/opsin_inverse.h"
#include "pik/single_image_handler.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

// Verifies ReconOpsinImage reconstructs with low butteraugli distance.
void RoundTrip(const CompressParams& cparams, const FrameHeader& frame_header,
               const GroupHeader& header, const Image3F& opsin,
               Quantizer* quantizer, const ColorCorrelationMap& cmap,
               const Rect& cmap_rect, const CodecInOut* io0, ThreadPool* pool) {
  AcStrategyImage ac_strategy(quantizer->RawQuantField().xsize(),
                              quantizer->RawQuantField().ysize());
  FrameEncCache frame_enc_cache;
  frame_enc_cache.dequant_control_field = ImageB(
      DivCeil(opsin.xsize(), kTileDim), DivCeil(opsin.ysize(), kTileDim));
  ZeroFillImage(&frame_enc_cache.dequant_control_field);
  BlockDictionary dictionary;
  InitializeFrameEncCache(frame_header, opsin, ac_strategy, *quantizer, cmap,
                          dictionary, pool, &frame_enc_cache, nullptr);
  EncCache enc_cache;
  InitializeEncCache(frame_header, header, frame_enc_cache, Rect(opsin),
                     &enc_cache);
  enc_cache.ac_strategy = ac_strategy.Copy();
  ComputeCoefficients(*quantizer, cmap, cmap_rect, pool, frame_enc_cache,
                      &enc_cache);

  FrameDecCache frame_dec_cache;
  frame_dec_cache.dequant_control_field = ImageB(
      DivCeil(opsin.xsize(), kTileDim), DivCeil(opsin.ysize(), kTileDim));
  ZeroFillImage(&frame_dec_cache.dequant_control_field);
  frame_dec_cache.dc = CopyImage(frame_enc_cache.dc_dec);
  frame_dec_cache.gradient = std::move(frame_enc_cache.gradient);
  if (frame_header.flags & FrameHeader::kGradientMap) {
    ApplyGradientMap(frame_dec_cache.gradient, *quantizer, &frame_dec_cache.dc);
  } else {
    AdaptiveDCReconstruction(frame_dec_cache.dc, *quantizer, pool);
  }
  // Override quant field with the one seen by decoder.
  frame_dec_cache.raw_quant_field = std::move(enc_cache.quant_field);
  frame_dec_cache.ac_strategy = std::move(enc_cache.ac_strategy);

  GroupDecCache group_dec_cache;

  InitializeDecCache(frame_dec_cache, Rect(opsin), &group_dec_cache);
  DequantImageAC(*quantizer, cmap, cmap_rect, enc_cache.ac, &frame_dec_cache,
                 &group_dec_cache, Rect(opsin), /*aux_out=*/nullptr);

  Image3F recon(frame_dec_cache.dc.xsize() * kBlockDim,
                frame_dec_cache.dc.ysize() * kBlockDim);
  ReconOpsinImage(frame_header, header, *quantizer, Rect(enc_cache.quant_field),
                  &frame_dec_cache, &group_dec_cache, &recon, Rect(recon));

  PIK_CHECK(FinalizeFrameDecoding(&recon, io0->xsize(), io0->ysize(),
                                  frame_header, NoiseParams(), *quantizer,
                                  dictionary, pool, &frame_dec_cache));
  CodecInOut io1(io0->Context());
  io1.SetFromImage(std::move(recon), io0->Context()->c_linear_srgb[0]);

  EXPECT_LE(ButteraugliDistance(io0, &io1, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.2);
}

void RunRGBRoundTrip(float distance, bool fast) {
  CodecContext codec_context;
  ThreadPool pool(4);

  const std::string& pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  constexpr int N = kBlockDim;
  CodecInOut io(&codec_context);
  PIK_CHECK(io.SetFromFile(pathname, &pool));

  Image3F opsin =
      PadImageToMultiple(OpsinDynamicsImage(&io, Rect(io.color())), N);
  opsin = GaborishInverse(opsin, 1.0);

  ColorCorrelationMap cmap(opsin.xsize(), opsin.ysize());
  const Rect cmap_rect(cmap.ytob_map);
  DequantMatrices dequant(/*need_inv_matrices=*/true);
  Quantizer quantizer(&dequant, opsin.xsize() / N, opsin.ysize() / N);
  quantizer.SetQuant(4.0f);

  CompressParams cparams;
  cparams.butteraugli_distance = distance;
  cparams.fast_mode = fast;

  FrameHeader frame_header;
  GroupHeader header;
  frame_header.gaborish = GaborishStrength::k1000;

  RoundTrip(cparams, frame_header, header, opsin, &quantizer, cmap, cmap_rect,
            &io, &pool);
}

TEST(CompressedImageTest, RGBRoundTrip_1) { RunRGBRoundTrip(1.0, false); }

TEST(CompressedImageTest, RGBRoundTrip_1_fast) { RunRGBRoundTrip(1.0, true); }

TEST(CompressedImageTest, RGBRoundTrip_2) { RunRGBRoundTrip(2.0, false); }

TEST(CompressedImageTest, RGBRoundTrip_2_fast) { RunRGBRoundTrip(2.0, true); }

}  // namespace
}  // namespace pik
