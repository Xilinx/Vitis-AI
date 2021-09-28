// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COMPRESSED_IMAGE_H_
#define PIK_COMPRESSED_IMAGE_H_

#include <stddef.h>
#include <stdint.h>

#include "pik/adaptive_reconstruction.h"
#include "pik/bit_reader.h"
#include "pik/block_dictionary.h"
#include "pik/color_correlation.h"
#include "pik/common.h"
#include "pik/compressed_image_fwd.h"
#include "pik/data_parallel.h"
#include "pik/headers.h"
#include "pik/image.h"
#include "pik/multipass_handler.h"
#include "pik/noise.h"
#include "pik/padded_bytes.h"
#include "pik/pik_info.h"
#include "pik/pik_params.h"
#include "pik/quantizer.h"

#include <ap_int.h>
#include <hls_stream.h>

// Methods to encode (decode) an image into (from) the bit stream:
// initialization of per-pass information and per-group information, actual
// computation of quantized coefficients, and encoding, plus corresponding
// methods for the decoder.

namespace pik {

struct GradientMap;

// Initialize per-pass information.
SIMD_ATTR void InitializeFrameEncCache(
    const FrameHeader& frame_header, const Image3F& opsin_full,
    const AcStrategyImage& ac_strategy, const Quantizer& quantizer,
    const ColorCorrelationMap& cmap, const BlockDictionary& dictionary,
    FrameEncCache* frame_enc_cache, PikInfo* aux_out);

// Initializes the encoder cache, setting parameters from the headers,
// setting up the `coeffs` and `dc_init` images in enc_cache.
SIMD_ATTR void InitializeEncCache(const FrameHeader& frame_header,
                                  const GroupHeader& group_header,
                                  const FrameEncCache& frame_enc_cache,
                                  const Rect& group_rect, EncCache* enc_cache);

// Computes quantized coefficients from the non-quantized ones already present
// in enc_cache.
SIMD_ATTR void ComputeCoefficients(const Quantizer& quantizer,
                                   const ColorCorrelationMap& cmap,
                                   const Rect& cmap_rect,
                                   const FrameEncCache& frame_enc_cache,
                                   EncCache* enc_cache,
                                   PikInfo* aux_out = nullptr);

// Encodes AC quantized coefficients from the given encoder cache.
PaddedBytes EncodeToBitstream(const EncCache& cache, const Rect& rect,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params, bool fast_mode,
                              MultipassHandler* handler,
                              PikInfo* info = nullptr);

// Encodes AC quantized coefficients from the given encoder cache.
PaddedBytes hls_EncodeToBitstream(const EncCache& cache, const Rect& rect,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params, bool fast_mode,
                              MultipassHandler* handler,
                              PikInfo* info = nullptr,
							  ap_uint<32> *ac_x=0,
							  ap_uint<32> *ac_y=0,
							  ap_uint<32> *ac_b=0,
							  ap_uint<32> *k2_order=0);

// Decodes AC coefficients from the bit stream, populating the AC
// fields of the decoder cache, and the corresponding rectangles in the global
// information (quant_field and ac_strategy) in the per-pass decoder cache.
template <bool first>
bool DecodeFromBitstream(const FrameHeader& frame_header,
                         const GroupHeader& header,
                         const PaddedBytes& compressed, BitReader* reader,
                         const Rect& group_rect, MultipassHandler* handler,
                         const size_t xsize_blocks, const size_t ysize_blocks,
                         const ColorCorrelationMap& cmap, const Rect& cmap_rect,
                         NoiseParams* noise_params, const Quantizer& quantizer,
                         FrameDecCache* PIK_RESTRICT frame_dec_cache,
                         GroupDecCache* PIK_RESTRICT group_dec_cache,
                         PikInfo* aux_out);

// Dequantizes the provided quantized_ac image into the decoder cache. Used in
// the encoder loop in adaptive_quantization.cc
void DequantImageAC(const Quantizer& quantizer, const ColorCorrelationMap& cmap,
                    const Rect& cmap_rect, const Image3S& quantized_ac,
                    FrameDecCache* PIK_RESTRICT frame_dec_cache,
                    GroupDecCache* PIK_RESTRICT group_dec_cache,
                    const Rect& group_rect, PikInfo* aux_out);

// Applies predictions to de-quantized AC coefficients, copies DC coefficients
// into AC, and does IDCT. Writes opsin IDCT values into `idct:idct_rect`.
void ReconOpsinImage(const FrameHeader& frame_header, const GroupHeader& header,
                     const Quantizer& quantizer, const Rect& block_group_rect,
                     FrameDecCache* PIK_RESTRICT frame_dec_cache,
                     GroupDecCache* PIK_RESTRICT group_dec_cache,
                     Image3F* PIK_RESTRICT idct, const Rect& idct_rect,
                     PikInfo* aux_out = nullptr, size_t downsample = 1);

// Finalizes the decoding of a pass by running per-pass post processing:
// smoothing and adaptive reconstruction. Writes linear sRGB to `idct` and
// shrinks it to `x/ysize` to undo prior padding.
// TODO(janwas): move NoiseParams into FrameHeader.
Status FinalizeFrameDecoding(Image3F* PIK_RESTRICT idct, size_t xsize,
                             size_t ysize, const FrameHeader& frame_header,
                             const NoiseParams& noise_params,
                             const Quantizer& quantizer,
                             const BlockDictionary& block_dictionary,
                             FrameDecCache* frame_dec_cache,
                             PikInfo* aux_out = nullptr, size_t downsample = 1);

}  // namespace pik

#endif  // PIK_COMPRESSED_IMAGE_H_
