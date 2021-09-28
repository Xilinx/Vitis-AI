// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/alpha.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <memory>
#include <vector>

#include "pik/ans_decode.h"
#include "pik/bit_reader.h"
#include "pik/dc_predictor.h"
#include "pik/entropy_coder.h"
#include "pik/fast_log.h"
#include "pik/lossless16.h"
#include "pik/lossless8.h"
#include "pik/profiler.h"
#include "pik/status.h"
#include "pik/write_bits.h"

namespace pik {

namespace {

// TODO(veluca): check if those upper bounds can be improved.
const constexpr int kRleSymStart[2] = {10, 18};

const constexpr int kMaxRleBits = 31;
const constexpr int kMaxRleLength =
    DecodeVarLenUint(kMaxRleBits, (1u << kMaxRleBits) - 1);

}  // namespace

Status EncodeAlpha(const CompressParams& params, const ImageU& plane,
                   const Rect& rect, int bit_depth, Alpha* alpha) {
  PIK_ASSERT(bit_depth == 8 || bit_depth == 16);
  alpha->bytes_per_alpha = bit_depth / 8;  // The encoding format used
  ImageS alpha_img(rect.xsize(), rect.ysize());
  if (alpha->bytes_per_alpha == 2) {
    for (size_t y = 0; y < rect.ysize(); y++) {
      int16_t* PIK_RESTRICT row = alpha_img.Row(y);
      const uint16_t* PIK_RESTRICT in = rect.ConstRow(plane, y);
      for (size_t x = 0; x < rect.xsize(); x++) {
        row[x] = in[x] - (1 << 15);
      }
    }
  } else {
    for (size_t y = 0; y < rect.ysize(); y++) {
      int16_t* PIK_RESTRICT row = alpha_img.Row(y);
      const uint16_t* PIK_RESTRICT in = rect.ConstRow(plane, y);
      for (size_t x = 0; x < rect.xsize(); x++) {
        row[x] = in[x];
      }
    }
  }
  ImageS residuals(rect.xsize(), rect.ysize());
  ShrinkY(Rect(alpha_img), alpha_img, Rect(residuals), &residuals);
  std::string best;

  const size_t rle_sym_start = kRleSymStart[alpha->bytes_per_alpha - 1];

  for (bool rle : {true, false}) {
    std::vector<std::vector<Token>> tokens(1);

    size_t cnt = 0;

    auto encode_cnt = [&]() {
      if (cnt > 0) {
        int nbits, bits;
        EncodeVarLenUint(cnt - 1, &nbits, &bits);
        tokens[0].emplace_back(Token(0, rle_sym_start + nbits, nbits, bits));
        cnt = 0;
      }
    };
    for (size_t y = 0; y < residuals.ysize(); y++) {
      const int16_t* PIK_RESTRICT row = residuals.ConstRow(y);
      for (size_t x = 0; x < residuals.xsize(); x++) {
        if (!rle || row[x]) {
          encode_cnt();
          int nbits, bits;
          EncodeVarLenInt(row[x], &nbits, &bits);
          PIK_ASSERT(nbits < rle_sym_start);
          tokens[0].emplace_back(Token(0, nbits, nbits, bits));
        } else {
          if (++cnt == kMaxRleLength) {
            encode_cnt();
          }
        }
      }
    }
    encode_cnt();

    std::vector<ANSEncodingData> codes;
    std::vector<uint8_t> context_map;
    std::string enc =
        BuildAndEncodeHistograms(1, tokens, &codes, &context_map, nullptr);
    enc += WriteTokens(tokens[0], codes, context_map, nullptr);
    if (best.empty() || best.size() > enc.size()) {
      best = std::move(enc);
    }
  }
  alpha->encoded.resize(best.size());
  memcpy(alpha->encoded.data(), best.data(), best.size());
  return true;
}

Status DecodeAlpha(const DecompressParams& params, const Alpha& alpha,
                   ImageU* plane, const Rect& rect) {
  PROFILER_FUNC;
  PIK_CHECK(plane->xsize() != 0);
  if (alpha.bytes_per_alpha != 1 && alpha.bytes_per_alpha != 2) {
    return PIK_FAILURE("Invalid bytes_per_alpha");
  }

  BitReader bit_reader(alpha.encoded.data(), alpha.encoded.size());
  std::vector<uint8_t> context_map;
  ANSCode code;
  PIK_RETURN_IF_ERROR(
      DecodeHistograms(&bit_reader, 1, 54, &code, &context_map));
  ANSSymbolReader decoder(&code);

  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  const size_t rle_sym_start = kRleSymStart[alpha.bytes_per_alpha - 1];

  ImageS residuals(xsize, ysize);
  const int histo_idx = context_map[0];
  size_t skip = 0;
  for (size_t y = 0; y < ysize; y++) {
    int16_t* PIK_RESTRICT row = residuals.Row(y);
    for (size_t x = 0; x < xsize; x++) {
      if (skip) {
        row[x] = 0;
        skip--;
        continue;
      }
      bit_reader.FillBitBuffer();
      int s = decoder.ReadSymbol(histo_idx, &bit_reader);
      if (s > 0) {
        if (s >= rle_sym_start) {
          s -= rle_sym_start;
          if (s > kMaxRleBits) {
            return PIK_FAILURE("Invalid rle nbits");
          }
          int bits = bit_reader.ReadBits(s);
          s = DecodeVarLenUint(s, bits);
          skip = s;
          row[x] = 0;
          continue;
        }
        int bits = bit_reader.ReadBits(s);
        s = DecodeVarLenInt(s, bits);
      }
      row[x] = s;
    }
  }
  if (skip != 0) {
    return PIK_FAILURE("Invalid alpha");
  }
  PIK_RETURN_IF_ERROR(bit_reader.JumpToByteBoundary());
  ImageS alpha_img(xsize, ysize);
  ExpandY(Rect(alpha_img), residuals, &alpha_img);
  if (alpha.bytes_per_alpha == 2) {
    for (size_t y = 0; y < ysize; y++) {
      const int16_t* PIK_RESTRICT in = alpha_img.ConstRow(y);
      uint16_t* PIK_RESTRICT row = rect.Row(plane, y);
      for (size_t x = 0; x < rect.xsize(); x++) {
        row[x] = in[x] + (1 << 15);
      }
    }
  } else {
    for (size_t y = 0; y < ysize; y++) {
      const int16_t* PIK_RESTRICT in = alpha_img.ConstRow(y);
      uint16_t* PIK_RESTRICT row = rect.Row(plane, y);
      for (size_t x = 0; x < rect.xsize(); x++) {
        row[x] = in[x];
      }
    }
  }

  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}

}  // namespace pik
