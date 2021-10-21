// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/pik_frame.h"

#include "pik/status.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits.h> // PATH_MAX
#include <limits>
#include <memory>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/ac_strategy.h"
#include "pik/adaptive_quantization.h"
#include "pik/alpha.h"
#include "pik/ar_control_field.h"
#include "pik/arch_specific.h"
#include "pik/bilinear_transform.h"
#include "pik/bit_reader.h"
#include "pik/bits.h"
#include "pik/byte_order.h"
#include "pik/color_correlation.h"
#include "pik/color_encoding.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/compressed_dc.h"
#include "pik/convolve.h"
#include "pik/dct.h"
#include "pik/dct_util.h"
#include "pik/entropy_coder.h"
#include "pik/external_image.h"
#include "pik/fast_log.h"
#include "pik/gaborish.h"
#include "pik/gamma_correct.h"
#include "pik/headers.h"
#include "pik/image.h"
#include "pik/lossless16.h"
#include "pik/lossless8.h"
#include "pik/multipass_handler.h"
#include "pik/noise.h"
#include "pik/opsin_image.h"
#include "pik/opsin_inverse.h"
#include "pik/padded_bytes.h"
#include "pik/pik_params.h"
#include "pik/profiler.h"
#include "pik/resize.h"
#include "pik/simd/targets.h"
#include "pik/size_coder.h"

#include "XAccPIKKernel1.hpp"
#include "XAccPIKKernel2.hpp"
#include "XAccPIKKernel3.hpp"
#include "host_dev.hpp"
#include "kernel3/encode_order.hpp"

#include <ap_int.h>
#include <hls_stream.h>

namespace pik {
namespace {

// For encoder.
uint32_t FrameFlagsFromParams(const CompressParams& cparams, const CodecInOut* io) {
    uint32_t flags = 0;

    const float dist = cparams.butteraugli_distance;

    // We don't add noise at low butteraugli distances because the original
    // noise is stored within the compressed image and adding noise makes things
    // worse.
    if (ApplyOverride(cparams.noise, dist >= kMinButteraugliForNoise)) {
        flags |= FrameHeader::kNoise;
    }

    if (ApplyOverride(cparams.gradient, dist >= kMinButteraugliForGradient)) {
        flags |= FrameHeader::kGradientMap;
    }

    if (io->IsGray()) {
        flags |= FrameHeader::kGrayscaleOpt;
    }

    return flags;
}

void OverrideFlag(const Override o, const uint32_t flag, uint32_t* PIK_RESTRICT flags) {
    if (o == Override::kOn) {
        *flags |= flag;
    } else if (o == Override::kOff) {
        *flags &= ~flag;
    }
}

void OverridePassFlags(const DecompressParams& dparams, FrameHeader* PIK_RESTRICT frame_header) {
    OverrideFlag(dparams.noise, FrameHeader::kNoise, &frame_header->flags);
    OverrideFlag(dparams.gradient, FrameHeader::kGradientMap, &frame_header->flags);

    if (dparams.adaptive_reconstruction == Override::kOff) {
        frame_header->have_adaptive_reconstruction = false;
    } else if (dparams.adaptive_reconstruction == Override::kOn) {
        frame_header->have_adaptive_reconstruction = true;
    }
    frame_header->epf_params.use_sharpened =
        ApplyOverride(dparams.epf_use_sharpened, frame_header->epf_params.use_sharpened);
    if (dparams.epf_sigma > 0) {
        frame_header->epf_params.enable_adaptive = false;
        frame_header->epf_params.sigma = dparams.epf_sigma;
    }

    if (dparams.gaborish != -1) {
        frame_header->gaborish = GaborishStrength(dparams.gaborish);
    }
}

void OverrideGroupFlags(const DecompressParams& dparams,
                        const FrameHeader* PIK_RESTRICT frame_header,
                        GroupHeader* PIK_RESTRICT header) {}

// Specializes a 8-bit and 16-bit of rounding from floating point to lossless.
template <typename T>
T RoundForLossless(float in);

template <>
uint8_t RoundForLossless(float in) {
    // NOTE: if in was originally an 8 or 16 bit value, we don't need to round
    // because such values are exactly representable as floats. Rounding is only
    // needed when forcing inexact values back to integers.
    return static_cast<uint8_t>(in + 0.5f);
}

template <>
uint16_t RoundForLossless(float in) {
    return static_cast<uint16_t>(in * 257.0f + 0.5f);
}

// Specializes a 8-bit and 16-bit lossless diff for previous pass.
template <typename T>
T DiffForLossless(float in, float prev);

template <>
uint8_t DiffForLossless(float in, float prev) {
    uint8_t diff = static_cast<int>(RoundForLossless<uint8_t>(in)) - static_cast<int>(RoundForLossless<uint8_t>(prev));
    if (diff > 127)
        diff = (255 - diff) * 2 + 1;
    else
        diff = diff * 2;
    return diff;
}

template <>
uint16_t DiffForLossless(float in, float prev) {
    uint32_t diff = 0xFFFF & (static_cast<int>(RoundForLossless<uint16_t>(in)) -
                              static_cast<int>(RoundForLossless<uint16_t>(prev)));
    if (diff > 32767)
        diff = (65535 - diff) * 2 + 1;
    else
        diff = diff * 2;
    return diff;
}

// Handles one channel c for converting ImageF or Image3F to lossless 8-bit or
// lossless 16-bit, and optionally handles previous pass delta.
template <typename T>
void LosslessChannelPass(
    const int c, const CodecInOut* io, const Rect& rect, const Image3F& previous_pass, Image<T>* channel_out) {
    size_t xsize = rect.xsize();
    size_t ysize = rect.ysize();
    if (previous_pass.xsize() == 0) {
        for (size_t y = 0; y < ysize; ++y) {
            const float* const PIK_RESTRICT row_in = rect.ConstPlaneRow(io->color(), c, y);
            T* const PIK_RESTRICT row_out = channel_out->Row(y);
            for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = RoundForLossless<T>(row_in[x]);
            }
        }
    } else {
        for (size_t y = 0; y < ysize; ++y) {
            const float* const PIK_RESTRICT row_in = rect.ConstPlaneRow(io->color(), c, y);
            T* const PIK_RESTRICT row_out = channel_out->Row(y);
            const float* const PIK_RESTRICT row_prev = previous_pass.PlaneRow(0, y);
            for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = DiffForLossless<T>(row_in[x], row_prev[x]);
            }
        }
    }
}

Status PixelsToPikLosslessFrame(CompressParams cparams,
                                const FrameHeader& frame_header,
                                const CodecInOut* io,
                                const Rect& rect,
                                const Image3F& previous_pass,
                                PaddedBytes* compressed,
                                size_t& pos,
                                PikInfo* aux_out) {
    PIK_ASSERT(pos % kBitsPerByte == 0);
    size_t xsize = rect.xsize();
    size_t ysize = rect.ysize();
    if (frame_header.lossless_grayscale) {
        if (frame_header.lossless_16_bits) {
            ImageU channel(xsize, ysize);
            LosslessChannelPass(0, io, rect, previous_pass, &channel);
            compressed->resize(pos / kBitsPerByte);
            if (!Grayscale16bit_compress(channel, compressed)) {
                return PIK_FAILURE("Lossless compression failed");
            }
        } else {
            ImageB channel(xsize, ysize);
            LosslessChannelPass(0, io, rect, previous_pass, &channel);
            compressed->resize(pos / kBitsPerByte);
            if (!Grayscale8bit_compress(channel, compressed)) {
                return PIK_FAILURE("Lossless compression failed");
            }
        }
    } else {
        if (frame_header.lossless_16_bits) {
            Image3U image(xsize, ysize);
            LosslessChannelPass(0, io, rect, previous_pass, const_cast<ImageU*>(&image.Plane(0)));
            LosslessChannelPass(1, io, rect, previous_pass, const_cast<ImageU*>(&image.Plane(1)));
            LosslessChannelPass(2, io, rect, previous_pass, const_cast<ImageU*>(&image.Plane(2)));
            compressed->resize(pos / kBitsPerByte);
            if (!Colorful16bit_compress(image, compressed)) {
                return PIK_FAILURE("Lossless compression failed");
            }
        } else {
            Image3B image(xsize, ysize);
            LosslessChannelPass(0, io, rect, previous_pass, const_cast<ImageB*>(&image.Plane(0)));
            LosslessChannelPass(1, io, rect, previous_pass, const_cast<ImageB*>(&image.Plane(1)));
            LosslessChannelPass(2, io, rect, previous_pass, const_cast<ImageB*>(&image.Plane(2)));
            compressed->resize(pos / kBitsPerByte);
            if (!Colorful8bit_compress(image, compressed)) {
                return PIK_FAILURE("Lossless compression failed");
            }
        }
    }
    pos = compressed->size() * kBitsPerByte;
    return true;
}

// Returns the target size based on whether bitrate or direct targetsize is
// given.
size_t TargetSize(const CompressParams& cparams, const Rect& rect) {
    if (cparams.target_size > 0) {
        return cparams.target_size;
    }
    if (cparams.target_bitrate > 0.0) {
        return 0.5 + cparams.target_bitrate * rect.xsize() * rect.ysize() / 8;
    }
    return 0;
}

Status PikPassHeuristics(CompressParams cparams,
                         const FrameHeader& frame_header,
                         const Image3F& opsin_orig,
                         const Image3F& opsin,
                         DequantMatrices* dequant,
                         ImageB* dequant_control_field,
                         uint8_t dequant_map[kMaxQuantControlFieldValue][256],
                         MultipassManager* multipass_manager,
                         GroupHeader* template_group_header,
                         ColorCorrelationMap* full_cmap,
                         std::shared_ptr<Quantizer>* full_quantizer,
                         AcStrategyImage* full_ac_strategy,
                         ImageB* full_ar_sigma_lut_ids,
                         BlockDictionary* block_dictionary,
                         PikInfo* aux_out) {
    size_t target_size = TargetSize(cparams, Rect(opsin_orig));
    // TODO(robryk): This should take *template_group_header size, and size of
    // other passes into account.
    size_t opsin_target_size = target_size;
    if (cparams.target_size > 0 || cparams.target_bitrate > 0.0) {
        cparams.target_size = opsin_target_size;
    } else if (cparams.butteraugli_distance < 0) {
        return PIK_FAILURE("Expected non-negative distance");
    }

    template_group_header->nonserialized_have_alpha = frame_header.has_alpha;

    if (cparams.lossless_mode) {
        return true;
    }

    constexpr size_t N = kBlockDim;
    PROFILER_ZONE("enc OpsinToPik uninstrumented");
    const size_t xsize = opsin_orig.xsize();
    const size_t ysize = opsin_orig.ysize();
    const size_t xsize_blocks = DivCeil(xsize, N);
    const size_t ysize_blocks = DivCeil(ysize, N);

    ImageF quant_field = InitialQuantField(cparams.butteraugli_distance, cparams.GetIntensityMultiplier(), opsin_orig,
                                           cparams, /*pool=*/nullptr, 1.0);

    //  *block_dictionary = multipass_manager->GetBlockDictionary(
    //      cparams.butteraugli_distance, opsin);

    //  Image3F opsin_with_removed_blocks = CopyImage(opsin);
    //  block_dictionary->SubtractFromDict(&opsin_with_removed_blocks);
    //  ApplyReverseBilinear(&opsin_with_removed_blocks);

    /*  multipass_manager->GetDequantMatrices(
          cparams.butteraugli_distance, cparams.GetIntensityMultiplier(),
          opsin_with_removed_blocks, quant_field, dequant, dequant_control_field,
          dequant_map);*/

    *dequant = DequantMatrices(/*need_inv_matrices=*/true);
    *dequant_control_field = ImageB(DivCeil(opsin.xsize(), kTileDim), DivCeil(opsin.ysize(), kTileDim));
    ZeroFillImage(dequant_control_field);
    memset(dequant_map, 0, kMaxQuantControlFieldValue * 256);

    multipass_manager->GetColorCorrelationMap(opsin, dequant, &*full_cmap);

    multipass_manager->GetAcStrategy(cparams.butteraugli_distance, &quant_field, dequant, opsin,
                                     /*pool=*/nullptr, full_ac_strategy, aux_out);

    // TODO(veluca): investigate if this should be included in
    // multipass_manager.
    FindBestArControlField(cparams.butteraugli_distance, cparams.GetIntensityMultiplier(), opsin, *full_ac_strategy,
                           quant_field, dequant, frame_header.gaborish,
                           /*pool=*/nullptr, full_ar_sigma_lut_ids);

    *full_quantizer = multipass_manager->GetQuantizer(
        cparams, xsize_blocks, ysize_blocks, opsin_orig, opsin, frame_header, *template_group_header, *full_cmap,
        *block_dictionary, *full_ac_strategy, *full_ar_sigma_lut_ids, dequant, *dequant_control_field, dequant_map,
        quant_field, aux_out);
    return true;
}

void strmToString(const int num_in, hls::stream<uint8_t>& strm_in, std::string& output) {
    for (int j = 0; j < num_in; ++j) {
        output[j] = strm_in.read();
    }
    output.resize(num_in);
}
// maybe parallelize color
std::string hls_EncodeCoeffOrders_top(const int32_t order[3][64]) { //, hls_PikImageSizeInfo &order_info

    std::string encoded_coeff_order(3 * 1024, 0);
    uint8_t* storage = reinterpret_cast<uint8_t*>(&encoded_coeff_order[0]);
    int storage_ix = 0;

    int num_bits = 0; // pos
    int num_pair = 0;

    hls::stream<nbits_t> strm_nbits;
    hls::stream<uint16_t> strm_bits("order_strm_bits");
    hls::stream<uint8_t> strm_order_byte("order_byte");
    hls::stream<bool> strm_order_e("strm_order_e");

    int hls_pos = 0;
    uint8_t tail_bits = 0;
    hls::stream<int> strm_order;

    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < 64; j++) {
            strm_order.write(order[c][j]);
        }

        hls_EncodeCoeffOrder(strm_order, num_bits, num_pair, strm_nbits, strm_bits);

        hls_WriteBitToStream(num_pair, tail_bits, strm_nbits, strm_bits, hls_pos, strm_order_byte, strm_order_e);

        _XF_IMAGE_PRINT("--byte_tail = %d , pos=%d\n", (int)tail_bits, (int)hls_pos);
        _XF_IMAGE_PRINT("--num_pair = %d \n", num_pair);
    }

    while (!strm_order_e.empty()) strm_order_e.read();

    if (hls_pos & (7)) {
        strm_order_byte.write(tail_bits);
    }

    // 4. Close the order bit stream.
    _XF_IMAGE_PRINT("storage_ix=%d \n", (int)hls_pos);

    strmToString(((hls_pos + 7) >> 3), strm_order_byte, encoded_coeff_order);
    encoded_coeff_order.resize((hls_pos + 7) >> 3);
    return encoded_coeff_order;
}

inline void XAcc_EncodeFloatParam(float val, float precision, size_t* storage_ix, uint8_t* storage) {
    WriteBits(1, val >= 0 ? 1 : 0, storage_ix, storage);
    const int absval_quant = static_cast<int>(std::abs(val) * precision + 0.5f);
    PIK_ASSERT(absval_quant < (1 << 16));
    WriteBits(16, absval_quant, storage_ix, storage);
}

void XAcc_EncodeNoise(const NoiseParams& noise_params, uint8_t storage[hls_kMaxNoiseSize], uint8_t& storage_size) {
#pragma HLS INLINE
    const size_t kMaxNoiseSize = 16;
    const float kNoisePrecision = 1000.0f;
    // std::string output(kMaxNoiseSize, 0);
    // uint8_t output[kMaxNoiseSize];
    size_t storage_ix = 0;
    // uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
    // uint8_t storage[kMaxNoiseSize];
    storage[0] = 0;
    const bool have_noise = (noise_params.alpha != 0.0f || noise_params.gamma != 0.0f || noise_params.beta != 0.0f);
    WriteBits(1, have_noise, &storage_ix, storage);
    if (have_noise) {
        XAcc_EncodeFloatParam(noise_params.alpha, kNoisePrecision, &storage_ix, storage);
        XAcc_EncodeFloatParam(noise_params.gamma, kNoisePrecision, &storage_ix, storage);
        XAcc_EncodeFloatParam(noise_params.beta, kNoisePrecision, &storage_ix, storage);
    }
    size_t jump_bits = ((storage_ix + 7) & ~7) - storage_ix;
    WriteBits(jump_bits, 0, &storage_ix, storage);
    PIK_ASSERT(storage_ix % 8 == 0);
    storage_size = storage_ix >> 3;
    // output.resize(output_size);
    // return output;
}

void arrayCharToString(const int num_in, ap_uint<32>* array_in, std::string& output) {
    for (int j = 0; j < num_in; ++j) {
        output[j] = (ap_uint<8>)array_in[j];
    }
    output.resize(num_in);
}

void arrayShortToString(const int num_in, ap_uint<32>* array_in, std::string& output) {
    std::cout << "shortTnt:" << std::endl;
    for (int i = 0; i < num_in; i += 2) {
        ap_uint<32> shortInt = array_in[i >> 1];
        output[i] = shortInt(7, 0);
        output[i + 1] = shortInt(15, 8);
        std::cout << std::dec << "," << shortInt;
    }
    std::cout << std::endl;
    output.resize(num_in);
}

Status PixelsToPikGroup(CompressParams cparams,
                        const FrameHeader& frame_header,
                        GroupHeader header,
                        const AcStrategyImage& ac_strategy,
                        const Quantizer* full_quantizer,
                        const ColorCorrelationMap& full_cmap,
                        const CodecInOut* io,
                        const Image3F& opsin_in,
                        const NoiseParams& noise_params,
                        size_t& pos,
                        const FrameEncCache& frame_enc_cache,
                        PikInfo* aux_out,
                        EncCache* cache,
                        MultipassHandler* multipass_handler) {
    const Rect& rect = multipass_handler->GroupRect();
    const Rect& padded_rect = multipass_handler->PaddedGroupRect();
    const Rect area_to_encode = Rect(0, 0, padded_rect.xsize(), padded_rect.ysize());

    if (frame_header.has_alpha) {
        PROFILER_ZONE("enc alpha");
        PIK_RETURN_IF_ERROR(EncodeAlpha(cparams, io->alpha(), rect, io->AlphaBits(), &header.alpha));
    }
    header.nonserialized_have_alpha = frame_header.has_alpha;

    uint8_t compressed = 0;

    size_t extension_bits, total_bits;
    PIK_RETURN_IF_ERROR(CanEncode(header, &extension_bits, &total_bits));
    PIK_RETURN_IF_ERROR(WriteGroupHeader(header, extension_bits, &pos, (&compressed)));
    WriteZeroesToByteBoundary(&pos, (&compressed));
    if (aux_out != nullptr) {
        aux_out->layers[kLayerHeader].total_size += DivCeil(total_bits, kBitsPerByte);
    }

    if (cparams.lossless_mode) {
        // Done; we'll encode the entire image in one shot later.
        return true;
    }

    Rect group_in_color_tiles(multipass_handler->BlockGroupRect().x0() / kColorTileDimInBlocks,
                              multipass_handler->BlockGroupRect().y0() / kColorTileDimInBlocks,
                              DivCeil(multipass_handler->BlockGroupRect().xsize(), kColorTileDimInBlocks),
                              DivCeil(multipass_handler->BlockGroupRect().ysize(), kColorTileDimInBlocks));

    ColorCorrelationMap cmap = full_cmap.Copy(group_in_color_tiles);
    cache->saliency_threshold = cparams.saliency_threshold;
    cache->saliency_debug_skip_nonsalient = cparams.saliency_debug_skip_nonsalient;

    InitializeEncCache(frame_header, header, frame_enc_cache, multipass_handler->PaddedGroupRect(), cache);

    Quantizer quantizer = full_quantizer->Copy(multipass_handler->BlockGroupRect());

    ComputeCoefficients(quantizer, full_cmap, group_in_color_tiles, frame_enc_cache, cache, aux_out);

    printf("area_to_encode(%d,%d)\n", area_to_encode.x0(), area_to_encode.y0());

    return true;
}

// Max observed: 1.1M on RGB noise with d0.1.
// 512*512*4*2 = 2M should be enough for 16-bit RGBA images.
using GroupSizeCoder = SizeCoderT<0x150F0E0C>;

} // namespace

Status PixelsToPikPass(CompressParams cparams,
                       const FrameParams& frame_params,
                       const CodecInOut* io,
                       ThreadPool* pool,
                       PaddedBytes* compressed,
                       size_t& pos,
                       PikInfo* aux_out,
                       MultipassManager* multipass_manager) {
    FrameHeader frame_header;
    frame_header.num_passes = multipass_manager->GetNumPasses();
    frame_header.downsampling_factor_to_passes = multipass_manager->GetDownsamplingToNumPasses();
    frame_header.have_adaptive_reconstruction = false;
    if (cparams.lossless_mode) {
        frame_header.encoding = ImageEncoding::kLossless;
        frame_header.lossless_16_bits = io->original_bits_per_sample() > 8;
        frame_header.lossless_grayscale = io->IsGray();
    }

    frame_header.frame = frame_params.frame_info;
    frame_header.has_alpha = io->HasAlpha();

    if (frame_header.encoding == ImageEncoding::kPasses) {
        frame_header.flags = FrameFlagsFromParams(cparams, io);
        frame_header.predict_hf = cparams.predict_hf;
        frame_header.predict_lf = cparams.predict_lf;
        frame_header.gaborish = GaborishStrength(cparams.gaborish);

        if (ApplyOverride(cparams.adaptive_reconstruction,
                          cparams.butteraugli_distance >= kMinButteraugliForAdaptiveReconstruction)) {
            frame_header.have_adaptive_reconstruction = true;
            frame_header.epf_params.use_sharpened =
                ApplyOverride(cparams.epf_use_sharpened, frame_header.epf_params.use_sharpened);
            if (cparams.epf_sigma > 0) {
                frame_header.epf_params.enable_adaptive = false;
                frame_header.epf_params.sigma = cparams.epf_sigma;
            }
        }
    }

    multipass_manager->StartPass(frame_header);

    // TODO(veluca): delay writing the header until we know the total pass size.
    size_t extension_bits, total_bits;
    PIK_RETURN_IF_ERROR(CanEncode(frame_header, &extension_bits, &total_bits));
    compressed->resize(DivCeil(pos + total_bits, kBitsPerByte));
    PIK_RETURN_IF_ERROR(WritePassHeader(frame_header, extension_bits, &pos, compressed->data()));
    WriteZeroesToByteBoundary(&pos, compressed->data());
    if (aux_out != nullptr) {
        aux_out->layers[kLayerHeader].total_size += DivCeil(total_bits, kBitsPerByte);
    }

    const size_t xsize_groups = DivCeil(io->xsize(), kGroupDim);
    const size_t ysize_groups = DivCeil(io->ysize(), kGroupDim);
    const size_t num_groups = xsize_groups * ysize_groups;

    std::vector<MultipassHandler*> handlers(num_groups);
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
        const size_t gx = group_index % xsize_groups;
        const size_t gy = group_index / xsize_groups;
        const Rect rect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim, io->xsize(), io->ysize());
        handlers[group_index] = multipass_manager->GetGroupHandler(group_index, rect);
    }

    GroupHeader template_group_header;
    ColorCorrelationMap full_cmap(io->xsize(), io->ysize());
    std::shared_ptr<Quantizer> full_quantizer;
    AcStrategyImage full_ac_strategy;
    Image3F opsin_orig, opsin;
    NoiseParams noise_params;
    BlockDictionary block_dictionary;
    FrameEncCache frame_enc_cache;

    if (frame_header.encoding == ImageEncoding::kPasses) {
        opsin_orig = OpsinDynamicsImage(io, Rect(io->color()));
        if (aux_out != nullptr) {
            PIK_RETURN_IF_ERROR(aux_out->InspectImage3F("pik_pass:OpsinDynamicsImage", opsin_orig));
        }

        constexpr size_t N = kBlockDim;
        PROFILER_ZONE("enc OpsinToPik uninstrumented");
        const size_t xsize = opsin_orig.xsize();
        const size_t ysize = opsin_orig.ysize();
        if (xsize == 0 || ysize == 0) return PIK_FAILURE("Empty image");
        opsin = PadImageToMultiple(opsin_orig, N);

        if (frame_header.flags & FrameHeader::kNoise) {
            PROFILER_ZONE("enc GetNoiseParam");
            // Don't start at zero amplitude since adding noise is expensive -- it
            // significantly slows down decoding, and this is unlikely to completely
            // go away even with advanced optimizations. After the
            // kNoiseModelingRampUpDistanceRange we have reached the full level,
            // i.e. noise is no longer represented by the compressed image, so we
            // can add full noise by the noise modeling itself.
            static const double kNoiseModelingRampUpDistanceRange = 0.6;
            static const double kNoiseLevelAtStartOfRampUp = 0.25;
            // TODO(user) test and properly select quality_coef with smooth
            // filter
            float quality_coef = 1.0f;
            const double rampup =
                (cparams.butteraugli_distance - kMinButteraugliForNoise) / kNoiseModelingRampUpDistanceRange;
            if (rampup < 1.0) {
                quality_coef = kNoiseLevelAtStartOfRampUp + (1.0 - kNoiseLevelAtStartOfRampUp) * rampup;
            }
            GetNoiseParameter(opsin, &noise_params, quality_coef);
        }
        if (frame_header.gaborish != GaborishStrength::kOff) {
            opsin = GaborishInverse(opsin, 0.92718927264540152);
        }

        multipass_manager->DecorrelateOpsin(&opsin);

        PIK_RETURN_IF_ERROR(PikPassHeuristics(
            cparams, frame_header, opsin_orig, opsin, &frame_enc_cache.matrices, &frame_enc_cache.dequant_control_field,
            frame_enc_cache.dequant_map, multipass_manager, &template_group_header, &full_cmap, &full_quantizer,
            &full_ac_strategy, &frame_enc_cache.ar_sigma_lut_ids, &block_dictionary, aux_out));

        // Initialize frame_enc_cache and encode DC.
        InitializeFrameEncCache(frame_header, opsin, full_ac_strategy, *full_quantizer, full_cmap, block_dictionary,
                                &frame_enc_cache, aux_out);
        frame_enc_cache.use_new_dc = cparams.use_new_dc;

        PikImageSizeInfo* matrices_info = aux_out != nullptr ? &aux_out->layers[kLayerDequantTables] : nullptr;

        std::string dequant_code = frame_enc_cache.matrices.Encode(matrices_info);
        compressed->append(dequant_code);
        pos += dequant_code.size() * 8;
        std::cout << "dequant_code_pos=" << pos << std::endl;

        PaddedBytes pass_global_code;
        size_t byte_pos = 0;

        // Encode quantizer DC and global scale.
        PikImageSizeInfo* quant_info = aux_out ? &aux_out->layers[kLayerQuant] : nullptr;
        std::string quant_code = full_quantizer->Encode(quant_info);

        // Encode cmap. TODO(veluca): consider encoding DC part of cmap only here,
        // and AC in (super)groups.
        PikImageSizeInfo* cmap_info = aux_out ? &aux_out->layers[kLayerCmap] : nullptr;
        std::string cmap_code =
            EncodeColorMap(full_cmap.ytob_map, Rect(full_cmap.ytob_map), full_cmap.ytob_dc, cmap_info) +
            EncodeColorMap(full_cmap.ytox_map, Rect(full_cmap.ytox_map), full_cmap.ytox_dc, cmap_info);

        pass_global_code.resize(quant_code.size() + cmap_code.size());
        Append(quant_code, &pass_global_code, &byte_pos);
        Append(cmap_code, &pass_global_code, &byte_pos);

        PikImageSizeInfo* dc_info = aux_out != nullptr ? &aux_out->layers[kLayerDC] : nullptr;
        PikImageSizeInfo* cfields_info = aux_out != nullptr ? &aux_out->layers[kLayerControlFields] : nullptr;

        pass_global_code.append(EncodeDCGroups(*full_quantizer, frame_enc_cache, full_ac_strategy, multipass_manager,
                                               dc_info, cfields_info));
        compressed->append(pass_global_code);
        pos += pass_global_code.size() * 8;
        std::cout << "pass_global_code_pos=" << pos << std::endl;

        PikImageSizeInfo* dictionary_info = aux_out ? &aux_out->layers[kLayerDictionary] : nullptr;
        std::string dictionary_code = block_dictionary.Encode(dictionary_info);
        compressed->append(dictionary_code);
        pos += dictionary_code.size() * 8;
        std::cout << "dictionary_code_pos=" << pos << std::endl;

        std::string quant_cf_code = EncodeDequantControlField(frame_enc_cache.dequant_control_field, matrices_info);
        quant_cf_code +=
            EncodeDequantControlFieldMap(full_quantizer->RawQuantField(), frame_enc_cache.dequant_control_field,
                                         frame_enc_cache.dequant_map, matrices_info);
        compressed->append(quant_cf_code);
        pos += quant_cf_code.size() * 8;
        std::cout << "quant_cf_code_pos=" << pos << std::endl;
    }

    // Compress groups: one per combination of group and pass. Outer loop lists
    // passes, inner lists groups. Group headers are only encoded in the groups
    // of the first pass.
    std::vector<std::vector<PaddedBytes> > group_codes(num_groups);
    std::atomic<int> num_errors{0};
    for (int group_index = 0; group_index < num_groups; ++group_index) {
        std::vector<PaddedBytes>* group_code = &group_codes[group_index];
        size_t group_pos = 0;
        group_code->resize(multipass_manager->GetNumPasses());
        /*
        if (!PixelsToPikGroup(cparams, frame_header, template_group_header,
                              full_ac_strategy, full_quantizer.get(), full_cmap, io,
                              opsin, noise_params, group_code, group_pos,
                              frame_enc_cache, aux_out, handlers[group_index])) {
          num_errors.fetch_add(1, std::memory_order_relaxed);

          continue;
        }
        */
    };

    for (size_t i = 0; i < num_groups; i++) {
        PIK_ASSERT(group_codes[i].size() == multipass_manager->GetNumPasses());
    }

    PIK_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

    // Build TOCs.

    for (size_t i = 0; i < multipass_manager->GetNumPasses(); i++) {
        size_t group_toc_pos = 0;
        PaddedBytes group_toc(PaddedBytes(GroupSizeCoder::MaxSize(num_groups)));
        uint8_t* group_toc_storage = group_toc.data();
        size_t total_groups_size = 0;
        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            size_t group_size = group_codes[group_index][i].size();
            GroupSizeCoder::Encode(group_size, &group_toc_pos, group_toc_storage);
            total_groups_size += group_size;
        }
        WriteZeroesToByteBoundary(&group_toc_pos, group_toc_storage);
        group_toc.resize(group_toc_pos / kBitsPerByte);

        // Push output.
        PIK_ASSERT(pos % kBitsPerByte == 0);
        compressed->reserve(DivCeil(pos, kBitsPerByte) + group_toc.size() + total_groups_size);
        compressed->append(group_toc);
        pos += group_toc.size() * kBitsPerByte;
        std::cout << "group_toc_pos=" << pos << std::endl;

        // Only do lossless encoding in the first pass, if there is more than one.
        if (frame_header.encoding == ImageEncoding::kLossless && i == 0) {
            // Encode entire image at once to avoid per-group overhead. Must come
            // BEFORE the encoded groups because the decoder assumes that the last
            // group coincides with the end of the bitstream.
            const Rect rect(io->color());

            Image3F previous_pass;
            PIK_RETURN_IF_ERROR(multipass_manager->GetPreviousPass(io->dec_c_original, pool, &previous_pass));
            PIK_RETURN_IF_ERROR(
                PixelsToPikLosslessFrame(cparams, frame_header, io, rect, previous_pass, compressed, pos, aux_out));
        }

        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            const PaddedBytes& group_code = group_codes[group_index][i];
            compressed->append(group_code);
            pos += group_code.size() * kBitsPerByte;
            std::cout << "group_code_pos=" << pos << std::endl;
        }
    }

    io->enc_size = compressed->size();
    return true;
}

void Exp(ImageF* out) {
    for (int y = 0; y < out->ysize(); ++y) {
        float* const PIK_RESTRICT row_out = out->Row(y);
        for (int x = 0; x < out->xsize(); ++x) {
            row_out[x] = exp(row_out[x]);
        }
    }
}

using DCGroupSizeCoder = SizeCoderT<0x150F0E0C>;
using GroupSizeCoder = SizeCoderT<0x150F0E0C>;

Status hls_PixelsToPikPass(CompressParams cparams,
                           std::string xclbinPath,
                           const FrameParams& frame_params,
                           const CodecInOut* io,
                           ThreadPool* pool,
                           PaddedBytes* compressed,
                           size_t& pos,
                           PikInfo* aux_out,
                           MultipassManager* multipass_manager) {
    FrameHeader frame_header;
    frame_header.num_passes = multipass_manager->GetNumPasses();
    frame_header.downsampling_factor_to_passes = multipass_manager->GetDownsamplingToNumPasses();
    frame_header.have_adaptive_reconstruction = false;
    if (cparams.lossless_mode) {
        frame_header.encoding = ImageEncoding::kLossless;
        frame_header.lossless_16_bits = io->original_bits_per_sample() > 8;
        frame_header.lossless_grayscale = io->IsGray();
    }

    frame_header.frame = frame_params.frame_info;
    frame_header.has_alpha = io->HasAlpha();

    if (frame_header.encoding == ImageEncoding::kPasses) {
        frame_header.flags = FrameFlagsFromParams(cparams, io);
        frame_header.predict_hf = cparams.predict_hf;
        frame_header.predict_lf = cparams.predict_lf;
        frame_header.gaborish = GaborishStrength(cparams.gaborish);

        if (ApplyOverride(cparams.adaptive_reconstruction,
                          cparams.butteraugli_distance >= kMinButteraugliForAdaptiveReconstruction)) {
            frame_header.have_adaptive_reconstruction = true;
            frame_header.epf_params.use_sharpened =
                ApplyOverride(cparams.epf_use_sharpened, frame_header.epf_params.use_sharpened);
            if (cparams.epf_sigma > 0) {
                frame_header.epf_params.enable_adaptive = false;
                frame_header.epf_params.sigma = cparams.epf_sigma;
            }
        }
    }

    multipass_manager->StartPass(frame_header);

    // TODO(veluca): delay writing the header until we know the total pass size.
    size_t extension_bits, total_bits;
    PIK_RETURN_IF_ERROR(CanEncode(frame_header, &extension_bits, &total_bits));
    compressed->resize(DivCeil(pos + total_bits, kBitsPerByte));
    PIK_RETURN_IF_ERROR(WritePassHeader(frame_header, extension_bits, &pos, compressed->data()));
    WriteZeroesToByteBoundary(&pos, compressed->data());
    if (aux_out != nullptr) {
        aux_out->layers[kLayerHeader].total_size += DivCeil(total_bits, kBitsPerByte);
    }

    const size_t xsize_groups = DivCeil(io->xsize(), kGroupDim);
    const size_t ysize_groups = DivCeil(io->ysize(), kGroupDim);
    const size_t num_groups = xsize_groups * ysize_groups;

    std::vector<MultipassHandler*> handlers(num_groups);
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
        const size_t gx = group_index % xsize_groups;
        const size_t gy = group_index / xsize_groups;
        const Rect rect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim, io->xsize(), io->ysize());
        handlers[group_index] = multipass_manager->GetGroupHandler(group_index, rect);
    }

    GroupHeader template_group_header;
    ColorCorrelationMap full_cmap(io->xsize(), io->ysize());
    std::shared_ptr<Quantizer> full_quantizer;
    AcStrategyImage full_ac_strategy;
    //  Image3F opsin_orig, opsin;
    NoiseParams noise_params;
    BlockDictionary block_dictionary;
    FrameEncCache frame_enc_cache;

    constexpr size_t N = kBlockDim;
    const size_t xsizet = Rect(io->color()).xsize();
    const size_t ysizet = Rect(io->color()).ysize();
    const size_t xsize_blocks = DivCeil(xsizet, N);
    const size_t ysize_blocks = DivCeil(ysizet, N);
    const size_t ysize_padded = ysize_blocks * kBlockDim;
    const size_t xsize_padded = xsize_blocks * kBlockDim;
    Image3F opsin_orig(xsizet, ysizet);
    Image3F opsin(xsize_padded, ysize_padded);

    //============================kernel1 initialize==========================

    // initial kernel1 buffer and config
    const Image3F* linear_srgb = &io->color();
    Image3F copy;
    Rect linear_rect = Rect(io->color());
    if (!io->IsLinearSRGB()) {
        const ColorEncoding& c = io->Context()->c_linear_srgb[io->IsGray()];
        PIK_CHECK(io->CopyTo(Rect(io->color()), c, &copy));
        linear_srgb = &copy;
        // We've cut out the rectangle, start at x0=y0=0 in copy.
        linear_rect = Rect(copy);
    }

    size_t target_size = TargetSize(cparams, Rect(opsin_orig));
    size_t opsin_target_size = target_size;
    if (cparams.target_size > 0 || cparams.target_bitrate > 0.0) {
        cparams.target_size = opsin_target_size;
    } else if (cparams.butteraugli_distance < 0) {
        return PIK_FAILURE("Expected non-negative distance");
    }

    ap_uint<32>* axi_out = (ap_uint<32>*)malloc(8192 * 8192 * 3 * sizeof(ap_uint<32>));
    ap_uint<32>* axi_cmap = (ap_uint<32>*)malloc((128 * 128 * 2 + 2) * sizeof(ap_uint<32>));
    ap_uint<32>* axi_qf = (ap_uint<32>*)malloc((1024 * 1024 + 2) * sizeof(ap_uint<32>));
    hls::stream<float> rgbStrm[3];
    hls::stream<float> xybStrm[3];
    hls::stream<float> xybGabStrm[3];
    hls::stream<float> yOrigStrm("yorig");
    hls::stream<DT> ostrm[3];
    hls::stream<bool> e_ostrm[3];

    const int nums = xsizet * ysizet;
    int len[3] = {nums * DT_SZ, nums * DT_SZ, nums * DT_SZ};
    int offset[3] = {0, nums * DT_SZ, 2 * nums * DT_SZ};
    float* buf = (float*)malloc(BUF_DEPTH * sizeof(float));
    memset(buf, 0, sizeof(float) * BUF_DEPTH);

    DT* const d0ptr = (DT*)(buf);
    DT* ptr = d0ptr;
    for (int y = 0; y < ysizet; ++y) {
        const float* PIK_RESTRICT row_in0 = linear_rect.ConstPlaneRow(*linear_srgb, 0, y);
        memcpy(ptr, row_in0, xsizet * sizeof(float));
        ptr = ptr + xsizet;
    }

    DT* const d1ptr = (DT*)(buf + nums);
    ptr = d1ptr;
    for (int y = 0; y < ysizet; ++y) {
        const float* PIK_RESTRICT row_in1 = linear_rect.ConstPlaneRow(*linear_srgb, 1, y);
        memcpy(ptr, row_in1, xsizet * sizeof(float));
        ptr = ptr + xsizet;
    }

    DT* const d2ptr = (DT*)(buf + 2 * nums);
    ptr = d2ptr;
    for (int y = 0; y < ysizet; ++y) {
        const float* PIK_RESTRICT row_in2 = linear_rect.ConstPlaneRow(*linear_srgb, 2, y);
        memcpy(ptr, row_in2, xsizet * sizeof(float));
        ptr = ptr + xsizet;
    }

    static const float kAcQuant = 0.97136686727219523;
    const float intensity_multiplier3 = std::cbrt(cparams.GetIntensityMultiplier());
    const float quant_ac = intensity_multiplier3 * kAcQuant / cparams.butteraugli_distance;

    ap_uint<32> k1_config[32];
    k1_config[0] = len[0];
    k1_config[1] = len[1];
    k1_config[2] = len[2];
    k1_config[3] = offset[0];
    k1_config[4] = offset[1];
    k1_config[5] = offset[2];
    k1_config[6] = xsizet;
    k1_config[7] = ysizet;
    k1_config[8] = fToBits<float, int32_t>(quant_ac);
    k1_config[9] = nums;
    k1_config[10] = 3 * nums;

#ifdef HLS_TEST
    kernel1Top(k1_config, (ap_uint<AXI_WIDTH>*)buf, axi_out, axi_cmap, axi_qf);
#endif

    //=============================kernel1 end===========================

    //============================kernel2 initialize==========================

    /*
    uint32_t xsize;
    uint32_t ysize;
    uint32_t xblock8;
    uint32_t yblock8;
    uint32_t xblock32;
    uint32_t yblock32;
    uint32_t xgroup;
    uint32_t ygroup;

    int src_num;
    int in_quant_field_num;
    int cmap_num0;
    int cmap_num1;
    int ac_num;
    int dc_num;
    int acs_num;
    int out_quant_field_num;

    bool kChooseAcStrategy;
    float discretization_factor;
    float kMulInhomogeneity16x16;
    float kMulInhomogeneity32x32;
    float butteraugli_target;
    float intensity_multiplier;
    float quant_dc;
    */

    // initial kernel2 k2_config
    ap_uint<32>* ac = (ap_uint<32>*)malloc(3 * 8192 * 8192 * sizeof(ap_uint<32>));
    ap_uint<32>* dc = (ap_uint<32>*)malloc(3 * 1024 * 1024 * sizeof(ap_uint<32>));
    ap_uint<32>* quant_field_out = (ap_uint<32>*)malloc((1024 * 1024 + 2) * sizeof(ap_uint<32>));
    ap_uint<32>* ac_strategy = (ap_uint<32>*)malloc(1024 * 1024 * sizeof(ap_uint<32>));
    ap_uint<32>* block = (ap_uint<32>*)malloc(1024 * 1024 * sizeof(ap_uint<32>));
    ap_uint<32>* order = (ap_uint<32>*)malloc(64 * 16 * 16 * 3 * sizeof(ap_uint<32>));

    float butteraugli_target_dc =
        std::min<float>(cparams.butteraugli_distance, std::pow(cparams.butteraugli_distance, 0.57840232344431763));
    float quant_dc = intensity_multiplier3 * 0.74852919562896747 / butteraugli_target_dc;

    ap_uint<32> k2_config[32];

    k2_config[0] = xsize_padded;
    k2_config[1] = ysize_padded;
    k2_config[2] = xsize_padded / 8;
    k2_config[3] = ysize_padded / 8;
    k2_config[4] = (k2_config[0] + 31) / 32;
    k2_config[5] = (k2_config[1] + 31) / 32;
    k2_config[6] = (k2_config[0] + 511) / 512;
    k2_config[7] = (k2_config[1] + 511) / 512;

    k2_config[8] = k2_config[4] * k2_config[5] * 3 * 32 * 32;
    k2_config[9] = k2_config[4] * k2_config[5] * 4 * 4 + 2;
    k2_config[10] = ((k2_config[0] + 63) / 64) * ((k2_config[1] + 63) / 64) * 2 + 2;
    k2_config[11] = ((k2_config[0] + 63) / 64) * ((k2_config[1] + 63) / 64);
    k2_config[12] = xsize_padded * ysize_padded;
    k2_config[13] = k2_config[2] * k2_config[3];
    k2_config[14] = k2_config[2] * k2_config[3];
    k2_config[15] = k2_config[2] * k2_config[3] + 2;

    k2_config[16] = true;
    k2_config[17] = fToBits<float, uint32_t>(100 * (6.9654004856811754) / cparams.butteraugli_distance);
    k2_config[18] = fToBits<float, uint32_t>((-47.780 * (3.9429727851421288)) / cparams.butteraugli_distance);
    k2_config[19] = fToBits<float, uint32_t>((-47.780 * (-4.270639713545533)) / cparams.butteraugli_distance);
    k2_config[20] = fToBits<float, uint32_t>(cparams.butteraugli_distance);
    k2_config[21] = fToBits<float, uint32_t>(intensity_multiplier3);
    k2_config[22] = fToBits<float, uint32_t>(quant_dc);
    k2_config[23] = k2_config[6] * k2_config[7] * 3 * 64;
    k2_config[24] = 3 * k2_config[13];

    k2_config[25] = k2_config[4] * k2_config[5] * 32 * 32 * 4;
    k2_config[26] = k2_config[25];
    k2_config[27] = k2_config[25];
    k2_config[28] = 0;
    k2_config[29] = k2_config[25];
    k2_config[30] = k2_config[25] * 2;

    int hls_ac_groups = num_groups;

#ifdef HLS_TEST
    kernel2Top(k2_config, (ap_uint<64>*)axi_out, axi_qf, axi_cmap, ac, dc, quant_field_out, ac_strategy, block, order);

    std::cout << "k2 order:" << std::endl;
    for (int i = 0; i < hls_ac_groups; i++) {
        for (int j = 0; j < 64 * 3; j++) {
            std::cout << (int)order[i * 3 * 64 + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 dc:" << std::endl;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < k2_config[3]; i++) {
            for (int j = 0; j < k2_config[2]; j++) {
                std::cout << (int)dc[c * MAX_NUM_BLOCK88 + i * k2_config[2] + j] << ",";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "k2 acs:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)ac_strategy[i * k2_config[2] + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 block:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)block[i * k2_config[2] + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 quant:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)quant_field_out[i * k2_config[2] + j] << ",";
        }
        std::cout << std::endl;
    }

#else
    std::cout << "k2 Config:" << std::endl;
    std::cout << "xsize:" << k2_config[0] << std::endl;
    std::cout << "ysize:" << k2_config[1] << std::endl;
    std::cout << "xblock8:" << k2_config[2] << std::endl;
    std::cout << "yblock8:" << k2_config[3] << std::endl;
    std::cout << "xblock32:" << k2_config[4] << std::endl;
    std::cout << "yblock32:" << k2_config[5] << std::endl;
    std::cout << "xgroup:" << k2_config[6] << std::endl;
    std::cout << "ygroup:" << k2_config[7] << std::endl;
    std::cout << "src_num:" << k2_config[8] << std::endl;
    std::cout << "in_quant_field_num:" << k2_config[9] << std::endl;
    std::cout << "cmap_num0:" << k2_config[10] << std::endl;
    std::cout << "cmap_num1:" << k2_config[11] << std::endl;
    std::cout << "ac_num:" << k2_config[12] << std::endl;
    std::cout << "dc_num:" << k2_config[13] << std::endl;
    std::cout << "acs_num:" << k2_config[14] << std::endl;
    std::cout << "out_quant_field_num:" << k2_config[15] << std::endl;
    std::cout << "choose_ac_astrategy:" << k2_config[16] << std::endl;
    std::cout << "discretization_factor:" << bitsToF<uint32_t, float>(k2_config[17]) << std::endl;
    std::cout << "kmul16:" << bitsToF<uint32_t, float>(k2_config[18]) << std::endl;
    std::cout << "kmul32:" << bitsToF<uint32_t, float>(k2_config[19]) << std::endl;
    std::cout << "butteraugli:" << bitsToF<uint32_t, float>(k2_config[20]) << std::endl;
    std::cout << "intensity_mul:" << bitsToF<uint32_t, float>(k2_config[21]) << std::endl;
    std::cout << "quant_dc:" << bitsToF<uint32_t, float>(k2_config[22]) << std::endl;
#endif

    //=============================kernel2 end============================

    //============================kernel3 initialize======================

    // initilaize kernel3 config
    ap_uint<32> k3_config[32];

    k3_config[0] = xsize_padded;
    k3_config[1] = ysize_padded;
    k3_config[2] = k3_config[0] / 8;
    k3_config[3] = k3_config[1] / 8;
    k3_config[4] = (k3_config[0] + 31) / 32;
    k3_config[5] = (k3_config[1] + 31) / 32;
    k3_config[6] = (k3_config[0] + 63) / 64;
    k3_config[7] = (k3_config[1] + 63) / 64;
    k3_config[8] = (k3_config[0] + 511) / 512;
    k3_config[9] = (k3_config[1] + 511) / 512;
    k3_config[10] = (k3_config[0] + 2047) / 2048;
    k3_config[11] = (k3_config[1] + 2047) / 2048;
    k3_config[12] = k3_config[8] * k3_config[9];
    k3_config[13] = k3_config[10] * k3_config[11];
    k3_config[14] = k3_config[2] * k3_config[3];
    k3_config[15] = k3_config[0] * k3_config[1] * 3;

    const size_t max_dc_histo_size = 1024 * (MAX_NUM_COLOR + 4);
    const size_t max_dc_size = 4 * (2 * hls_kTotalSize) + 4096; // to be test
    const size_t max_ac_size = 4 * (4 * hls_kTotalSize) + 4096;
    const size_t max_ac_histo_size = hls_kNumStaticContexts * 1024;

    ap_uint<32>* dc_histo_code_out = (ap_uint<32>*)malloc(2 * k3_config[13] * max_dc_histo_size * sizeof(ap_uint<32>));
    ap_uint<32>* dc_code_out = (ap_uint<32>*)malloc(2 * k3_config[13] * max_dc_size * sizeof(ap_uint<32>));
    ap_uint<32>* ac_histo_code_out = (ap_uint<32>*)malloc(k3_config[12] * max_ac_histo_size * sizeof(ap_uint<32>));
    ap_uint<32>* ac_code_out = (ap_uint<32>*)malloc(k3_config[12] * max_ac_size * sizeof(ap_uint<32>));

    int len_dc_histo[2 * MAX_DC_GROUP] = {0};
    int len_dc[2 * MAX_DC_GROUP] = {0};
    int len_ac_histo[MAX_AC_GROUP] = {0};
    int len_ac[MAX_AC_GROUP] = {0};
    ap_uint<32> histo_cfg[4 * MAX_DC_GROUP + 2 * MAX_AC_GROUP] = {0};

    memset(dc_histo_code_out, 0, 2 * k3_config[13] * max_dc_histo_size * sizeof(ap_uint<32>));
    memset(dc_code_out, 0, 2 * k3_config[13] * max_dc_size * sizeof(ap_uint<32>));
    memset(ac_histo_code_out, 0, k3_config[12] * max_ac_histo_size * sizeof(ap_uint<32>));
    memset(ac_code_out, 0, k3_config[12] * max_ac_size * sizeof(ap_uint<32>));

#ifdef HLS_TEST

    kernel3Top(k3_config, ac, dc, quant_field_out, ac_strategy, block, order, histo_cfg, dc_histo_code_out, dc_code_out,
               ac_histo_code_out, ac_code_out);

    for (int j = 0; j < 20; j++) {
        std::cout << ", " << (int)dc_histo_code_out[j];
    }
    std::cout << std::endl;
    for (int j = 0; j < 20; j++) {
        std::cout << ", " << (int)dc_histo_code_out[j + MAX_DC_HISTO_SIZE];
    }
    std::cout << std::endl;
    for (int j = 0; j < 20; j++) {
        std::cout << ", " << (int)dc_code_out[j];
    }
    std::cout << std::endl;
    for (int j = 0; j < 20; j++) {
        std::cout << ", " << (int)dc_code_out[j + MAX_DC_SIZE];
    }
    std::cout << std::endl;

    for (int j = 0; j < 2 * k3_config[13]; j++) {
        len_dc_histo[j] = histo_cfg[j];
        std::cout << "len_dc_h:" << (int)histo_cfg[j] << std::endl;
    }

    for (int j = 0; j < k3_config[12]; j++) {
        len_ac_histo[j] = histo_cfg[2 * k3_config[13] + j];
        std::cout << "len_ac_h:" << (int)histo_cfg[2 * k3_config[13] + j] << std::endl;
    }

    for (int j = 0; j < 2 * k3_config[13]; j++) {
        len_dc[j] = histo_cfg[2 * k3_config[13] + k3_config[12] + j];
        std::cout << "len_dc_c:" << (int)histo_cfg[2 * k3_config[13] + k3_config[12] + j] << std::endl;
    }

    for (int j = 0; j < k3_config[12]; j++) {
        len_ac[j] = histo_cfg[4 * k3_config[13] + k3_config[12] + j];
        std::cout << "len_ac_c:" << (int)histo_cfg[4 * k3_config[13] + k3_config[12] + j] << std::endl;
    }

    for (int j = 0; j < 20; j++) {
        std::cout << ", " << (int)ac_histo_code_out[j];
    }
    std::cout << std::endl;
    for (int j = 0; j < 20; j++) {
        std::cout << ", " << (int)ac_code_out[j];
    }
    std::cout << std::endl;
#endif

//============================kernel3 end=====================================

//==========================host code start===================================
#ifndef HLS_TEST
    std::cout << "openCL host start!" << std::endl;

    host_func(xclbinPath, buf, k1_config, k2_config, k3_config, axi_cmap, order, quant_field_out, len_dc_histo, len_dc,
              dc_histo_code_out, dc_code_out, len_ac_histo, len_ac, ac_histo_code_out, ac_code_out);

    std::cout << "openCL host end!" << std::endl;
#endif
//============================host code end===================================

// return kernel1 result to original code
#ifdef HLS_TEST

    int addr = 0;
    int x32 = (opsin.xsize() + 32 - 1) / 32;
    int y32 = (opsin.ysize() + 32 - 1) / 32;
    unsigned int xyb_int;
    for (int y = 0; y < y32 * 32; y = y + 32) {
        for (int x = 0; x < x32 * 32; x = x + 32) {
            for (int i = 0; i < 32; i++) {
                if (y + i < opsin.ysize()) {
                    float* PIK_RESTRICT row_xyb0 = opsin.PlaneRow(0, y + i);
                    for (int j = 0; j < 32; j++) {
                        if (x + j < opsin.xsize()) {
                            xyb_int = axi_out[addr].to_int();
                            row_xyb0[x + j] = bitsToF<unsigned int, float>(xyb_int);
                        }
                        addr++;
                    }
                } else {
                    for (int j = 0; j < 32; j++) {
                        addr++;
                    }
                }
            }

            for (int i = 0; i < 32; i++) {
                if (y + i < opsin.ysize()) {
                    float* PIK_RESTRICT row_xyb1 = opsin.PlaneRow(1, y + i);
                    for (int j = 0; j < 32; j++) {
                        if (x + j < opsin.xsize()) {
                            xyb_int = axi_out[addr].to_int();
                            row_xyb1[x + j] = bitsToF<unsigned int, float>(xyb_int);
                        }
                        addr++;
                    }
                } else {
                    for (int j = 0; j < 32; j++) {
                        addr++;
                    }
                }
            }

            for (int i = 0; i < 32; i++) {
                if (y + i < opsin.ysize()) {
                    float* PIK_RESTRICT row_xyb2 = opsin.PlaneRow(2, y + i);
                    for (int j = 0; j < 32; j++) {
                        if (x + j < opsin.xsize()) {
                            xyb_int = axi_out[addr].to_int();
                            row_xyb2[x + j] = bitsToF<unsigned int, float>(xyb_int);
                        }
                        addr++;
                    }
                } else {
                    for (int j = 0; j < 32; j++) {
                        addr++;
                    }
                }
            }
        }
    }

    static const int kResolution = 8;
    const size_t out_xsize = (xsizet + kResolution - 1) / kResolution;
    const size_t out_ysize = (ysizet + kResolution - 1) / kResolution;

    int cnt = 2;
    ImageF quant_field = ImageF(out_xsize, out_ysize);
    addr = 2;
    int x4 = (quant_field.xsize() + 4 - 1) / 4;
    int y4 = (quant_field.ysize() + 4 - 1) / 4;
    unsigned int qf_int;
    for (int y = 0; y < y4 * 4; y = y + 4) {
        for (int x = 0; x < x4 * 4; x = x + 4) {
            for (int i = 0; i < 4; i++) {
                if (y + i < quant_field.ysize()) {
                    float* PIK_RESTRICT row_qf = quant_field.Row(y + i);
                    for (int j = 0; j < 4; j++) {
                        if (x + j < quant_field.xsize()) {
                            qf_int = axi_qf[addr].to_int();
                            row_qf[x + j] = bitsToF<unsigned int, float>(qf_int);
                        }
                        addr++;
                    }
                } else {
                    for (int j = 0; j < 4; j++) {
                        addr++;
                    }
                }
            }
        }
    }

    qf_int = axi_qf[0].to_int();
    float avg = bitsToF<unsigned int, float>(qf_int);
    qf_int = axi_qf[1].to_int();
    float absavg = bitsToF<unsigned int, float>(qf_int);

#endif

    int cntCmap = 0;
    unsigned int cmap_int = axi_cmap[0].to_int();
    full_cmap.ytox_dc = cmap_int;
    cntCmap++;
    cmap_int = axi_cmap[1].to_int();
    full_cmap.ytob_dc = cmap_int;
    cntCmap++;
    for (int i = 0; i < full_cmap.ytob_map.ysize(); i++) {
        int* PIK_RESTRICT tmpb = full_cmap.ytob_map.Row(i);
        int* PIK_RESTRICT tmpx = full_cmap.ytox_map.Row(i);
        for (int j = 0; j < full_cmap.ytob_map.xsize(); j++) {
            cmap_int = axi_cmap[cntCmap].to_int();
            tmpx[j] = cmap_int;
            cntCmap++;
            cmap_int = axi_cmap[cntCmap].to_int();
            tmpb[j] = cmap_int;
            cntCmap++;
        }
    }
    //=============================return result==================================

    DequantMatrices* dequant = &frame_enc_cache.matrices;
    *dequant = DequantMatrices(/*need_inv_matrices=*/true);

#ifdef HLS_TEST
    multipass_manager->GetAcStrategy(cparams.butteraugli_distance, &quant_field, dequant, opsin,
                                     /*pool=*/nullptr, &full_ac_strategy, aux_out);
#endif

    ImageB* dequant_control_field = &frame_enc_cache.dequant_control_field;
    uint8_t dequant_map[kMaxQuantControlFieldValue][256];
    for (int i = 0; i < kMaxQuantControlFieldValue; i++) {
        for (int j = 0; j < 256; j++) {
            dequant_map[i][j] = frame_enc_cache.dequant_map[i][j];
        }
    }
    // TODO(veluca): investigate if this should be included in
    // multipass_manager.
    ImageB* full_ar_sigma_lut_ids = &frame_enc_cache.ar_sigma_lut_ids;

#ifdef HLS_TEST
    FindBestArControlField(cparams.butteraugli_distance, cparams.GetIntensityMultiplier(), opsin, full_ac_strategy,
                           quant_field, dequant, frame_header.gaborish,
                           /*pool=*/nullptr, full_ar_sigma_lut_ids);
#endif

    *dequant_control_field = ImageB(DivCeil(opsin.xsize(), kTileDim), DivCeil(opsin.ysize(), kTileDim));
    ZeroFillImage(dequant_control_field);
    memset(dequant_map, 0, kMaxQuantControlFieldValue * 256);

    template_group_header.nonserialized_have_alpha = frame_header.has_alpha;

#ifdef HLS_TEST

    full_quantizer = multipass_manager->GetQuantizerAvg(
        avg, absavg, cparams, xsize_blocks, ysize_blocks, opsin_orig, opsin, frame_header, template_group_header,
        full_cmap, block_dictionary, full_ac_strategy, *full_ar_sigma_lut_ids, dequant, *dequant_control_field,
        dequant_map, quant_field, aux_out);

    // Initialize frame_enc_cache and encode DC.
    InitializeFrameEncCache(frame_header, opsin, full_ac_strategy, *full_quantizer, full_cmap, block_dictionary,
                            &frame_enc_cache, aux_out);

#else

    full_quantizer = std::make_shared<Quantizer>(dequant, xsize_blocks, ysize_blocks);
    for (size_t by = 0; by < k2_config[3]; ++by) {
        int32_t* PIK_RESTRICT row_quant = full_quantizer->quant_img_ac_.Row(by);
        for (size_t bx = 0; bx < k2_config[2]; ++bx) {
            row_quant[bx] = quant_field_out[by * k2_config[2] + bx];
        }
    }

    full_quantizer->global_scale_ = 3065; // quant_field_out[k2_config[15]];
    full_quantizer->quant_dc_ = 16;       // quant_field_out[k2_config[15] + 1];

#endif

    frame_enc_cache.use_new_dc = cparams.use_new_dc;

    PikImageSizeInfo* matrices_info = aux_out != nullptr ? &aux_out->layers[kLayerDequantTables] : nullptr;

    std::string dequant_code = frame_enc_cache.matrices.Encode(matrices_info);
    compressed->append(dequant_code);
    pos += dequant_code.size() * 8;
    std::cout << "dequant_code:" << std::hex << dequant_code << std::endl;
    std::cout << "dequant_code_pos=" << pos << std::endl;

    PaddedBytes pass_global_code;
    size_t byte_pos = 0;

    // Encode quantizer DC and global scale.
    PikImageSizeInfo* quant_info = aux_out ? &aux_out->layers[kLayerQuant] : nullptr;
    std::string quant_code = full_quantizer->Encode(quant_info);
    std::cout << "quant_code:" << std::hex << quant_code << std::endl;

    // Encode cmap. TODO(veluca): consider encoding DC part of cmap only here,
    // and AC in (super)groups.
    PikImageSizeInfo* cmap_info = aux_out ? &aux_out->layers[kLayerCmap] : nullptr;
    std::string cmap_code = EncodeColorMap(full_cmap.ytob_map, Rect(full_cmap.ytob_map), full_cmap.ytob_dc, cmap_info) +
                            EncodeColorMap(full_cmap.ytox_map, Rect(full_cmap.ytox_map), full_cmap.ytox_dc, cmap_info);
    std::cout << "cmap_code:" << std::hex << cmap_code << std::endl;

    pass_global_code.resize(quant_code.size() + cmap_code.size());
    Append(quant_code, &pass_global_code, &byte_pos);
    Append(cmap_code, &pass_global_code, &byte_pos);

    PikImageSizeInfo* dictionary_info = aux_out ? &aux_out->layers[kLayerDictionary] : nullptr;
    std::string dictionary_code = block_dictionary.Encode(dictionary_info);
    std::cout << "dictionary_code:" << std::hex << dictionary_code << std::endl;

    std::string quant_cf_code = EncodeDequantControlField(frame_enc_cache.dequant_control_field, matrices_info);

    quant_cf_code += EncodeDequantControlFieldMap(full_quantizer->RawQuantField(),
                                                  frame_enc_cache.dequant_control_field, dequant_map, matrices_info);
    std::cout << "quant_cf_code:" << std::hex << quant_cf_code << std::endl;

    // Compress groups: one per combination of group and pass. Outer loop lists
    // passes, inner lists groups. Group headers are only encoded in the groups
    // of the first pass.
    std::atomic<int> num_errors{0};

    std::vector<PaddedBytes> group_codes(num_groups);
    std::vector<Image3S>* group_ac;
    std::vector<AcStrategyImage>* group_ac_strategy;
    std::vector<EncCache> ac_cache(num_groups);

#ifdef HLS_TEST

    for (int group_index = 0; group_index < num_groups; ++group_index) {
        size_t group_pos = 0;
        Image3S* hls_ac;
        AcStrategyImage* hls_ac_strategy;

        if (!PixelsToPikGroup(cparams, frame_header, template_group_header, full_ac_strategy, full_quantizer.get(),
                              full_cmap, io, opsin, noise_params, group_pos, frame_enc_cache, aux_out,
                              &(ac_cache[group_index]), handlers[group_index])) {
            num_errors.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
    };

#endif
    std::string compressed_dc[2 * MAX_DC_GROUP];
    int hls_dc_groups = k3_config[13];
    std::vector<PaddedBytes> group_codes_dc(hls_dc_groups);

    hls::stream<dct_t> strm_coef_raster_syn("strm_coef_raster_syn");
    ap_uint<32> hls_order[MAX_AC_GROUP][kOrderContexts][kDCTBlockSize];
    int order2enc[MAX_AC_GROUP][kOrderContexts][kDCTBlockSize];

    uint8_t noise_size[MAX_AC_GROUP];
    std::string noise_code[MAX_AC_GROUP];
    std::string order_code[MAX_AC_GROUP];
    std::string ac_histo_code[MAX_AC_GROUP];
    std::string ac_code[MAX_AC_GROUP];

    std::vector<Image3S> ddr_group_ac;

    for (int group_index = 0; group_index < num_groups; ++group_index) {
        Image3S& group_ac = ac_cache[group_index].ac;
        AcStrategyImage& group_ac_strategy = ac_cache[group_index].ac_strategy;

        PikImageSizeInfo* ac_info = aux_out != nullptr ? &aux_out->layers[kLayerAC] : nullptr; // clean to 0

        // EncodeNoise
        _XF_IMAGE_PRINT("EncodeNoise - E2B\n");

        uint8_t noise_out[hls_kMaxNoiseSize];
        XAcc_EncodeNoise(noise_params, noise_out, noise_size[group_index]);

        for (size_t i = 0; i < noise_size[group_index]; ++i)
            noise_code[group_index][i] = static_cast<char>(noise_out[i]);

        noise_code[group_index].resize(noise_size[group_index]);

        // EncodeCoeffOrders
        for (size_t c = 0; c < 3; c++) {
            for (size_t y = 0; y < 8; y++) {     // 8* 8
                for (size_t x = 0; x < 8; x++) { // 8
                    order2enc[group_index][c][y * 8 + x] = order[group_index * 3 * 64 + c * 64 + y * 8 + x];
                }
            }
        }

        _XF_IMAGE_PRINT("Enc(deCoeffOrders - E2B\n");
        order_code[group_index] = hls_EncodeCoeffOrders_top(order2enc[group_index]);
    } // end rect init

    std::string dc_histo_code[2 * MAX_DC_GROUP];
    std::string dc_code[2 * MAX_DC_GROUP];
    int offset_dc_histo = 0;
    int offset_dc = 0;
    for (int group_index = 0; group_index < (2 * hls_dc_groups); ++group_index) {
        dc_histo_code[group_index].resize(max_dc_histo_size);
        dc_code[group_index].resize(max_dc_size);

        arrayCharToString(len_dc_histo[group_index], (dc_histo_code_out + offset_dc_histo), dc_histo_code[group_index]);
        arrayShortToString(len_dc[group_index], (dc_code_out + offset_dc), dc_code[group_index]);
        compressed_dc[group_index] = dc_histo_code[group_index] + dc_code[group_index];

        offset_dc_histo += len_dc_histo[group_index];
        offset_dc += (len_dc[group_index] + 1) / 2;
    }

    int offset_ac_histo = 0;
    int offset_ac = 0;
    for (int group_index = 0; group_index < hls_ac_groups; ++group_index) {
        ac_histo_code[group_index].resize(max_ac_histo_size);
        ac_code[group_index].resize(max_ac_size);
        // std::cout<<"offset_ac:"<<offset_ac<<std::endl;
        arrayCharToString(len_ac_histo[group_index], (ac_histo_code_out + offset_ac_histo), ac_histo_code[group_index]);
        arrayShortToString(len_ac[group_index], (ac_code_out + offset_ac), ac_code[group_index]);

        offset_ac_histo += len_ac_histo[group_index];
        offset_ac += (len_ac[group_index] + 1) / 2;
    }

    std::vector<Token> ac_tokens; // no means
    for (int group_index = 0; group_index < hls_ac_groups; ++group_index) {
        PaddedBytes out(noise_size[group_index] + order_code[group_index].size() + ac_histo_code[group_index].size() +
                        ac_code[group_index].size());
        _XF_IMAGE_PRINT("noise_code size = %d\n", (int)noise_size[group_index]);
        _XF_IMAGE_PRINT("order_code size = %d\n", (int)order_code[group_index].size());
        _XF_IMAGE_PRINT("histo_code size = %d\n", (int)ac_histo_code[group_index].size());
        _XF_IMAGE_PRINT("ac_code size = %d\n", (int)ac_code[group_index].size());
        size_t byte_pos = 0;
        Append(noise_code[group_index], &out, &byte_pos);
        Append(order_code[group_index], &out, &byte_pos);
        Append(ac_histo_code[group_index], &out, &byte_pos);
        Append(ac_code[group_index], &out, &byte_pos);

        // TODO(veluca): fix this with DC supergroups.
        float output_size_estimate = out.size() - ac_code[group_index].size() - ac_histo_code[group_index].size();
        std::vector<std::array<size_t, 256> > counts(kNumContexts);
        size_t extra_bits = 0;
        for (const auto& token : ac_tokens) {
            counts[token.context][token.symbol]++;
            extra_bits += token.nbits;
        }
        float entropy_coded_bits = 0;
        for (size_t ctx = 0; ctx < kNumContexts; ctx++) {
            size_t total = std::accumulate(counts[ctx].begin(), counts[ctx].end(), size_t(0));
            if (total == 0) continue; // Prevent div by zero.
            double entropy = 0;
            for (size_t i = 0; i < 256; i++) {
                double p = 1.0 * counts[ctx][i] / total;
                if (p > 1e-4) {
                    entropy -= p * std::log(p);
                }
            }
            entropy_coded_bits += entropy * total / std::log(2);
        }
        output_size_estimate += static_cast<float>(extra_bits + entropy_coded_bits) / kBitsPerByte;
        if (aux_out != nullptr) aux_out->entropy_estimate = output_size_estimate;

        uint8_t tmp = 1;
        std::string header_str(1, 0);
        header_str[0] = static_cast<char>(tmp);
        PaddedBytes ac_header(1);
        byte_pos = 0;
        Append(header_str, &ac_header, &byte_pos);
        group_codes[group_index].append(ac_header);
        group_codes[group_index].append(out);
    } // end group ac

    _XF_IMAGE_PRINT("-Build TOCs!\n");

    for (int group_index = 0; group_index < hls_dc_groups; ++group_index) {
        size_t group_pos = 0;
        compressed_dc[group_index] = compressed_dc[2 * group_index] + compressed_dc[2 * group_index + 1];
        group_codes_dc[group_index].resize(compressed_dc[group_index].size());
        Append(compressed_dc[group_index], &group_codes_dc[group_index], &group_pos);
    }

    // Build TOCs.
    // TOC0+TOC1+...+TOCn+data0+...+datan
    PaddedBytes group_toc_dc(DCGroupSizeCoder::MaxSize(hls_dc_groups));
    size_t group_toc_pos_dc = 0;
    uint8_t* group_toc_storage_dc = group_toc_dc.data();
    size_t total_groups_size_dc = 0;

    //----------------
    for (size_t group_index = 0; group_index < hls_dc_groups; ++group_index) {
        size_t group_size = group_codes_dc[group_index].size();
        DCGroupSizeCoder::Encode(group_size, &group_toc_pos_dc, group_toc_storage_dc);
        _XF_IMAGE_PRINT("group_size= %d, group_toc_pos_dc=%d\n", (int)group_size, (int)group_toc_pos_dc);
        total_groups_size_dc += group_size;
    }
    //----------------
    _XF_IMAGE_PRINT("-start the group_toc/ code-DC group\n");
    WriteZeroesToByteBoundary(&group_toc_pos_dc, group_toc_storage_dc);
    group_toc_dc.resize(group_toc_pos_dc / kBitsPerByte);

    // Push output.
    PaddedBytes dc_codes;
    size_t pos_dc = 0;
    dc_codes.reserve(group_toc_dc.size() + total_groups_size_dc); //+serialized_gradient_map.size()
    dc_codes.append(group_toc_dc);
    pos_dc += group_toc_dc.size() * kBitsPerByte;
    //----------------
    for (size_t group_index = 0; group_index < hls_dc_groups; ++group_index) {
        const PaddedBytes& group_code = group_codes_dc[group_index];
        dc_codes.append(group_code);
        pos_dc += group_code.size() * kBitsPerByte;
    }
    std::cout << "dequant_code_pos=" << pos << std::endl;
    pass_global_code.append(dc_codes);
    compressed->append(pass_global_code);
    pos += pass_global_code.size() * 8;
    std::cout << "dequant_code_pos=" << pos << std::endl;

    compressed->append(dictionary_code);
    pos += dictionary_code.size() * 8;
    std::cout << "dequant_code_pos=" << pos << std::endl;

    compressed->append(quant_cf_code);
    pos += quant_cf_code.size() * 8;
    std::cout << "dequant_code_pos=" << pos << std::endl;

    for (size_t i = 0; i < multipass_manager->GetNumPasses(); i++) {
        size_t group_toc_pos = 0;
        PaddedBytes group_toc(PaddedBytes(GroupSizeCoder::MaxSize(num_groups)));
        uint8_t* group_toc_storage = group_toc.data();
        size_t total_groups_size = 0;

        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            size_t group_size = group_codes[group_index].size();
            _XF_IMAGE_PRINT("group_codes size = %d\n", (int)group_size);
            GroupSizeCoder::Encode(group_size, &group_toc_pos, group_toc_storage);
            total_groups_size += group_size;
        }
        WriteZeroesToByteBoundary(&group_toc_pos, group_toc_storage);
        group_toc.resize(group_toc_pos / kBitsPerByte);

        // Push output.
        PIK_ASSERT(pos % kBitsPerByte == 0);
        compressed->reserve(DivCeil(pos, kBitsPerByte) + group_toc.size() + total_groups_size);
        compressed->append(group_toc);
        pos += group_toc.size() * kBitsPerByte;
        std::cout << "dequant_code_pos=" << pos << std::endl;

        // Only do lossless encoding in the first pass, if there is more than one.
        if (frame_header.encoding == ImageEncoding::kLossless && i == 0) {
            // Encode entire image at once to avoid per-group overhead. Must come
            // BEFORE the encoded groups because the decoder assumes that the last
            // group coincides with the end of the bitstream.
            const Rect rect(io->color());

            Image3F previous_pass;
            PIK_RETURN_IF_ERROR(multipass_manager->GetPreviousPass(io->dec_c_original, pool, &previous_pass));
            PIK_RETURN_IF_ERROR(
                PixelsToPikLosslessFrame(cparams, frame_header, io, rect, previous_pass, compressed, pos, aux_out));
        }

        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            const PaddedBytes& group_code = group_codes[group_index];
            compressed->append(group_code);
            pos += group_code.size() * kBitsPerByte;
        }
    }

    io->enc_size = compressed->size();
    _XF_IMAGE_PRINT("compressed size = %d\n", (int)compressed->size());
    return true;
}

namespace {

Status ValidateImageDimensions(const FileHeader& file_header, const DecompressParams& dparams) {
    const size_t xsize = file_header.xsize();
    const size_t ysize = file_header.ysize();
    if (xsize == 0 || ysize == 0) {
        return PIK_FAILURE("Empty image.");
    }

    static const size_t kMaxWidth = (1 << 25) - 1;
    if (xsize > kMaxWidth) {
        return PIK_FAILURE("Image too wide.");
    }

    const size_t num_pixels = xsize * ysize;
    if (num_pixels > dparams.max_num_pixels) {
        return PIK_FAILURE("Image too big.");
    }

    return true;
}

// Specializes a 8-bit and 16-bit of converting to float from lossless.
float ToFloatForLossless(uint8_t in) {
    return static_cast<float>(in);
}

float ToFloatForLossless(uint16_t in) {
    return in * (1.0f / 257);
}

// Specializes a 8-bit and 16-bit undo of lossless diff.
float UndiffForLossless(uint8_t in, float prev) {
    uint16_t diff;
    if (in % 2 == 0)
        diff = in / 2;
    else
        diff = 255 - (in / 2);
    uint8_t out = diff + static_cast<int>(RoundForLossless<uint8_t>(prev));
    return ToFloatForLossless(out);
}

float UndiffForLossless(uint16_t in, float prev) {
    uint16_t diff;
    if (in % 2 == 0)
        diff = in / 2;
    else
        diff = 65535 - (in / 2);
    uint16_t out = diff + static_cast<int>(RoundForLossless<uint16_t>(prev));
    return ToFloatForLossless(out);
}

// Handles converting lossless 8-bit or lossless 16-bit, to Image3F, with
// option to give 3x same channel at input for grayscale, and optionally
// handles previous pass delta.
template <typename T>
void LosslessChannelDecodePass(
    int num_channels, const Image<T>** in, const Rect& rect, const Image3F& previous_pass, Image3F* color) {
    size_t xsize = rect.xsize();
    size_t ysize = rect.ysize();

    for (int c = 0; c < num_channels; c++) {
        if (previous_pass.xsize() == 0) {
            for (size_t y = 0; y < ysize; ++y) {
                const T* const PIK_RESTRICT row_in = in[c]->Row(y);
                float* const PIK_RESTRICT row_out = rect.PlaneRow(color, c, y);
                for (size_t x = 0; x < xsize; ++x) {
                    row_out[x] = ToFloatForLossless(row_in[x]);
                }
            }
        } else {
            for (size_t y = 0; y < ysize; ++y) {
                const T* const PIK_RESTRICT row_in = in[c]->Row(y);
                float* const PIK_RESTRICT row_out = rect.PlaneRow(color, c, y);
                const float* const PIK_RESTRICT row_prev = previous_pass.ConstPlaneRow(c, y);
                for (size_t x = 0; x < xsize; ++x) {
                    row_out[x] = UndiffForLossless(row_in[x], row_prev[x]);
                }
            }
        }
    }

    // Grayscale, copy the channel to the other two output channels
    if (num_channels == 1) {
        for (size_t y = 0; y < ysize; ++y) {
            const float* const PIK_RESTRICT row_0 = rect.PlaneRow(color, 0, y);
            float* const PIK_RESTRICT row_1 = rect.PlaneRow(color, 1, y);
            float* const PIK_RESTRICT row_2 = rect.PlaneRow(color, 2, y);
            for (size_t x = 0; x < xsize; ++x) {
                row_1[x] = row_2[x] = row_0[x];
            }
        }
    }
}

Status PikLosslessFrameToPixels(const PaddedBytes& compressed,
                                const FrameHeader& frame_header,
                                size_t* position,
                                Image3F* color,
                                const Rect& rect,
                                const Image3F& previous_pass) {
    PROFILER_FUNC;
    if (frame_header.lossless_grayscale) {
        if (frame_header.lossless_16_bits) {
            ImageU image;
            if (!Grayscale16bit_decompress(compressed, position, &image)) {
                return PIK_FAILURE("Lossless decompression failed");
            }
            if (!SameSize(image, rect)) {
                return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
            }
            const ImageU* array[1] = {&image};
            LosslessChannelDecodePass(1, array, rect, previous_pass, color);
        } else {
            ImageB image;
            if (!Grayscale8bit_decompress(compressed, position, &image)) {
                return PIK_FAILURE("Lossless decompression failed");
            }
            if (!SameSize(image, rect)) {
                return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
            }
            const ImageB* array[1] = {&image};
            LosslessChannelDecodePass(1, array, rect, previous_pass, color);
        }
    } else {
        if (frame_header.lossless_16_bits) {
            Image3U image;
            if (!Colorful16bit_decompress(compressed, position, &image)) {
                return PIK_FAILURE("Lossless decompression failed");
            }
            if (!SameSize(image, rect)) {
                return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
            }
            const ImageU* array[3] = {&image.Plane(0), &image.Plane(1), &image.Plane(2)};
            LosslessChannelDecodePass(3, array, rect, previous_pass, color);
        } else {
            Image3B image;
            if (!Colorful8bit_decompress(compressed, position, &image)) {
                return PIK_FAILURE("Lossless decompression failed");
            }
            if (!SameSize(image, rect)) {
                return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
            }
            const ImageB* array[3] = {&image.Plane(0), &image.Plane(1), &image.Plane(2)};
            LosslessChannelDecodePass(3, array, rect, previous_pass, color);
        }
    }
    return true;
}

// `reader` is a vector of readers (one per pass). Group headers are only
// present in the first pass, thus the group header of this group is read from
// `reader[0]`.
Status PikGroupToPixels(const DecompressParams& dparams,
                        const FileHeader& file_header,
                        const FrameHeader* frame_header,
                        const PaddedBytes& compressed,
                        const Quantizer& quantizer,
                        const ColorCorrelationMap& full_cmap,
                        std::vector<BitReader>* reader,
                        Image3F* PIK_RESTRICT opsin_output,
                        ImageU* alpha_output,
                        const CodecContext* context,
                        PikInfo* aux_out,
                        FrameDecCache* PIK_RESTRICT frame_dec_cache,
                        GroupDecCache* PIK_RESTRICT group_dec_cache,
                        MultipassHandler* multipass_handler,
                        const ColorEncoding& original_color_encoding,
                        size_t downsampling) {
    PROFILER_FUNC;
    const Rect& padded_rect = multipass_handler->PaddedGroupRect();
    const Rect& rect = multipass_handler->GroupRect();
    GroupHeader header;
    header.nonserialized_have_alpha = frame_header->has_alpha;
    PIK_RETURN_IF_ERROR(ReadGroupHeader(&(*reader)[0], &header));
    PIK_RETURN_IF_ERROR((*reader)[0].JumpToByteBoundary());
    OverrideGroupFlags(dparams, frame_header, &header);

    if (frame_header->has_alpha) {
        // TODO(lode): do not fail here based on the metadata
        // original_bytes_per_alpha, it should be allowed to use an efficient
        // encoding in pik which differs from what the original had (or
        // alternatively if they must be the same, there should not be two fields)
        if (header.alpha.bytes_per_alpha != file_header.metadata.transcoded.original_bytes_per_alpha) {
            return PIK_FAILURE("Nonuniform alpha bitdepth is not supported yet.");
        }
        if (file_header.metadata.transcoded.original_bytes_per_alpha == 0) {
            return PIK_FAILURE("Header claims to contain alpha but the depth is 0.");
        }
        PIK_RETURN_IF_ERROR(DecodeAlpha(dparams, header.alpha, alpha_output, rect));
    }

    if (frame_header->encoding == ImageEncoding::kLossless) {
        // Done; we'll decode the entire image in one shot later.
        return true;
    }

    ImageSize opsin_size = ImageSize::Make(padded_rect.xsize(), padded_rect.ysize());
    const size_t xsize_blocks = DivCeil<size_t>(opsin_size.xsize, kBlockDim);
    const size_t ysize_blocks = DivCeil<size_t>(opsin_size.ysize, kBlockDim);

    Rect group_in_color_tiles(multipass_handler->BlockGroupRect().x0() / kColorTileDimInBlocks,
                              multipass_handler->BlockGroupRect().y0() / kColorTileDimInBlocks,
                              DivCeil(multipass_handler->BlockGroupRect().xsize(), kColorTileDimInBlocks),
                              DivCeil(multipass_handler->BlockGroupRect().ysize(), kColorTileDimInBlocks));

    NoiseParams noise_params;

    InitializeDecCache(*frame_dec_cache, padded_rect, group_dec_cache);

    if (dparams.max_passes == 0) ZeroFillImage(&group_dec_cache->ac);
    for (size_t i = 0; i < frame_header->num_passes && i < dparams.max_passes; i++) {
        PROFILER_ZONE("dec_bitstr");
        auto decode = i == 0 ? &DecodeFromBitstream</*first=*/true> : &DecodeFromBitstream</*first=*/false>;
        if (!decode(*frame_header, header, compressed, &(*reader)[i], padded_rect, multipass_handler, xsize_blocks,
                    ysize_blocks, full_cmap, group_in_color_tiles, &noise_params, quantizer, frame_dec_cache,
                    group_dec_cache, aux_out)) {
            return PIK_FAILURE("Pik decoding failed.");
        }
        if (!(*reader)[i].JumpToByteBoundary()) {
            return PIK_FAILURE("Pik bitstream is corrupted.");
        }
    }

    Rect opsin_rect(padded_rect.x0() / downsampling, padded_rect.y0() / downsampling,
                    DivCeil(padded_rect.xsize(), downsampling), DivCeil(padded_rect.ysize(), downsampling));

    // Note: DecodeFromBitstream already performed dequantization.
    ReconOpsinImage(*frame_header, header, quantizer, multipass_handler->BlockGroupRect(), frame_dec_cache,
                    group_dec_cache, opsin_output, opsin_rect, aux_out, downsampling);

    return true;
}

} // namespace

Status PikPassToPixels(DecompressParams dparams,
                       const PaddedBytes& compressed,
                       const FileHeader& file_header,
                       ThreadPool* pool,
                       BitReader* reader,
                       CodecInOut* io,
                       PikInfo* aux_out,
                       MultipassManager* multipass_manager) {
    PROFILER_ZONE("PikPassToPixels uninstrumented");
    PIK_RETURN_IF_ERROR(ValidateImageDimensions(file_header, dparams));

    io->metadata = file_header.metadata;

    // Used when writing the output file unless DecoderHints overrides it.
    io->SetOriginalBitsPerSample(file_header.metadata.transcoded.original_bit_depth);
    io->dec_c_original = file_header.metadata.transcoded.original_color_encoding;
    if (io->dec_c_original.icc.empty()) {
        // Removed by MaybeRemoveProfile; fail unless we successfully restore it.
        PIK_RETURN_IF_ERROR(ColorManagement::SetProfileFromFields(&io->dec_c_original));
    }

    const size_t xsize = file_header.xsize();
    const size_t ysize = file_header.ysize();
    const size_t padded_xsize = DivCeil(xsize, kBlockDim) * kBlockDim;
    const size_t padded_ysize = DivCeil(ysize, kBlockDim) * kBlockDim;

    FrameHeader header;
    PIK_RETURN_IF_ERROR(ReadPassHeader(reader, &header));

    PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

    // TODO(veluca): add kProgressive.
    if (header.encoding != ImageEncoding::kPasses && header.encoding != ImageEncoding::kLossless) {
        return PIK_FAILURE("Unsupported bitstream");
    }

    OverridePassFlags(dparams, &header);

    size_t downsampling;
    if (dparams.max_downsampling >= 8) {
        downsampling = 8;
        dparams.max_passes = 0;
    } else {
        downsampling = 1;
        for (const auto& downsampling_and_num_passes : header.downsampling_factor_to_passes) {
            if (dparams.max_downsampling >= downsampling_and_num_passes.first &&
                dparams.max_passes > downsampling_and_num_passes.second) {
                downsampling = downsampling_and_num_passes.first;
                dparams.max_passes = downsampling_and_num_passes.second + 1;
            }
        }
    }
    if (aux_out != nullptr) {
        aux_out->downsampling = downsampling;
    }

    multipass_manager->StartPass(header);

    ImageU alpha;
    if (header.has_alpha) {
        alpha = ImageU(xsize, ysize);
    }

    const size_t xsize_groups = DivCeil(xsize, kGroupDim);
    const size_t ysize_groups = DivCeil(ysize, kGroupDim);
    const size_t num_groups = xsize_groups * ysize_groups;

    std::vector<PikInfo> aux_outs;
    if (aux_out != nullptr) {
        aux_outs.resize(num_groups, *aux_out);
    }
    std::vector<MultipassHandler*> handlers(num_groups);
    {
        PROFILER_ZONE("Get handlers");
        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            const size_t gx = group_index % xsize_groups;
            const size_t gy = group_index / xsize_groups;
            const size_t x = gx * kGroupDim;
            const size_t y = gy * kGroupDim;
            Rect rect(x, y, kGroupDim, kGroupDim, xsize, ysize);
            handlers[group_index] = multipass_manager->GetGroupHandler(group_index, rect);
        }
    }

    const size_t xsize_blocks = padded_xsize / kBlockDim;
    const size_t ysize_blocks = padded_ysize / kBlockDim;

    FrameDecCache frame_dec_cache;
    frame_dec_cache.use_new_dc = dparams.use_new_dc;
    frame_dec_cache.grayscale = header.flags & FrameHeader::kGrayscaleOpt;
    frame_dec_cache.ac_strategy = AcStrategyImage(xsize_blocks, ysize_blocks);
    frame_dec_cache.raw_quant_field = ImageI(xsize_blocks, ysize_blocks);
    frame_dec_cache.ar_sigma_lut_ids = ImageB(xsize_blocks, ysize_blocks);
    frame_dec_cache.dequant_control_field = ImageB(DivCeil(xsize, kTileDim), DivCeil(ysize, kTileDim));

    ColorCorrelationMap cmap(xsize, ysize);

    // TODO(veluca): deserialize quantization tables from the bitstream.

    Quantizer quantizer(&frame_dec_cache.matrices, 0, 0);
    BlockDictionary block_dictionary;

    std::vector<GroupDecCache> group_dec_caches(NumThreads(pool));

    if (header.encoding == ImageEncoding::kPasses) {
        PROFILER_ZONE("DecodeColorMap+DC");
        PIK_RETURN_IF_ERROR(frame_dec_cache.matrices.Decode(reader));
        PIK_RETURN_IF_ERROR(quantizer.Decode(reader));
        PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

        // TODO(veluca): decode quantization table mapping.

        DecodeColorMap(reader, &cmap.ytob_map, &cmap.ytob_dc);
        DecodeColorMap(reader, &cmap.ytox_map, &cmap.ytox_dc);
        PIK_RETURN_IF_ERROR(DecodeDCGroups(reader, compressed, header, xsize_blocks, ysize_blocks, quantizer, cmap,
                                           pool, multipass_manager, &frame_dec_cache, &group_dec_caches, aux_out));
        PIK_RETURN_IF_ERROR(block_dictionary.Decode(reader, padded_xsize, padded_ysize));

        // TODO(veluca): think of splitting this in DC groups.
        PIK_RETURN_IF_ERROR(DecodeDequantControlField(reader, &frame_dec_cache.dequant_control_field));
        PIK_RETURN_IF_ERROR(DecodeDequantControlFieldMap(reader, frame_dec_cache.raw_quant_field,
                                                         frame_dec_cache.dequant_control_field,
                                                         frame_dec_cache.dequant_map));
        multipass_manager->SaveAcStrategy(frame_dec_cache.ac_strategy);
        multipass_manager->SaveQuantField(frame_dec_cache.raw_quant_field);
    }

    Image3F opsin(DivCeil(padded_xsize, downsampling), DivCeil(padded_ysize, downsampling));

    // Read TOCs.
    std::vector<std::vector<size_t> > group_offsets(header.num_passes);
    std::vector<size_t> group_codes_begin(header.num_passes);
    for (size_t i = 0; i < header.num_passes; i++) {
        PROFILER_ZONE("Read TOC");
        std::vector<size_t>& group_offsets_pass = group_offsets[i];
        group_offsets_pass.reserve(num_groups + 1);
        group_offsets_pass.push_back(0);
        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            const uint32_t size = GroupSizeCoder::Decode(reader);
            group_offsets_pass.push_back(group_offsets_pass.back() + size);
        }
        PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());
        // On first pass, read lossless.
        if (header.encoding == ImageEncoding::kLossless && i == 0) {
            const Rect rect(0, 0, xsize, ysize);

            size_t pos = reader->Position();
            const size_t before_pos = pos;

            Image3F previous_pass;
            PIK_RETURN_IF_ERROR(multipass_manager->GetPreviousPass(io->dec_c_original, pool, &previous_pass));
            PIK_RETURN_IF_ERROR(PikLosslessFrameToPixels(compressed, header, &pos, &opsin, rect, previous_pass));
            reader->SkipBits((pos - before_pos) * kBitsPerByte);
            // Byte-wise; no need to jump to boundary.
        }
        // Pretend all groups of this pass are read.
        group_codes_begin[i] = reader->Position();
        reader->SkipBits(group_offsets_pass.back() * kBitsPerByte);
        if (reader->Position() > compressed.size()) {
            return PIK_FAILURE("Group code extends after stream end");
        }
    }

    // Decode groups.
    std::atomic<int> num_errors{0};
    const auto process_group = [&](const int group_index, const int thread) {
        std::vector<BitReader> readers;
        for (size_t i = 0; i < header.num_passes; i++) {
            size_t group_code_offset = group_offsets[i][group_index];
            size_t group_reader_limit = group_offsets[i][group_index + 1];
            // TODO(user): this looks ugly; we should get rid of PaddedBytes
            //               parameter once it is wrapped into BitReader; otherwise
            //               it is easy to screw the things up.
            readers.emplace_back(compressed.data(), group_codes_begin[i] + group_reader_limit);
            readers.back().SkipBits((group_codes_begin[i] + group_code_offset) * kBitsPerByte);
        }

        PikInfo* my_aux_out = aux_out ? &aux_outs[group_index] : nullptr;
        if (!PikGroupToPixels(dparams, file_header, &header, compressed, quantizer, cmap, &readers, &opsin, &alpha,
                              io->Context(), my_aux_out, &frame_dec_cache, &group_dec_caches[thread],
                              handlers[group_index], io->dec_c_original, downsampling)) {
            num_errors.fetch_add(1);
            return;
        }
    };
    {
        PROFILER_ZONE("PikPassToPixels pool");
        RunOnPool(pool, 0, num_groups, process_group, "PikPassToPixels");
    }

    PIK_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

    if (aux_out != nullptr) {
        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            aux_out->Assimilate(aux_outs[group_index]);
        }
    }

    if (header.encoding == ImageEncoding::kPasses) {
        multipass_manager->RestoreOpsin(&opsin);
        multipass_manager->SetDecodedPass(opsin);

        PIK_RETURN_IF_ERROR(FinalizeFrameDecoding(&opsin, file_header.xsize(), file_header.ysize(), header,
                                                  NoiseParams(), quantizer, block_dictionary, &frame_dec_cache, aux_out,
                                                  downsampling));
        // From now on, `opsin` is actually linear sRGB.

        if (header.flags & FrameHeader::kGrayscaleOpt) {
            PROFILER_ZONE("Grayscale opt");
            // Force all channels to gray
            for (size_t y = 0; y < opsin.ysize(); ++y) {
                float* PIK_RESTRICT row_r = opsin.PlaneRow(0, y);
                float* PIK_RESTRICT row_g = opsin.PlaneRow(1, y);
                float* PIK_RESTRICT row_b = opsin.PlaneRow(2, y);
                for (size_t x = 0; x < opsin.xsize(); x++) {
                    float gray = row_r[x] * 0.299 + row_g[x] * 0.587 + row_b[x] * 0.114;
                    row_r[x] = row_g[x] = row_b[x] = gray;
                }
            }
        }
        const ColorEncoding& c = io->Context()->c_linear_srgb[io->dec_c_original.IsGray()];
        io->SetFromImage(std::move(opsin), c);
    } else if (header.encoding == ImageEncoding::kLossless) {
        io->SetFromImage(std::move(opsin), io->dec_c_original);
        io->ShrinkTo(xsize, ysize);
        multipass_manager->SetDecodedPass(io);
    } else {
        return PIK_FAILURE("Unsupported image encoding");
    }

    if (header.has_alpha) {
        io->SetAlpha(std::move(alpha), 8 * file_header.metadata.transcoded.original_bytes_per_alpha);
    }

    io->ShrinkTo(DivCeil(xsize, downsampling), DivCeil(ysize, downsampling));

    return true;
}

} // namespace pik
