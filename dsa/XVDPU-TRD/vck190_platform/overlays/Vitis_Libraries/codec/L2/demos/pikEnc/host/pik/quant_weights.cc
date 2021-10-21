// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "pik/quant_weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "pik/bit_reader.h"
#include "pik/cache_aligned.h"
#include "pik/common.h"
#include "pik/dct.h"
#include "pik/huffman_decode.h"
#include "pik/huffman_encode.h"
#include "pik/image.h"
#include "pik/pik_info.h"
#include "pik/status.h"
#include "pik/write_bits.h"

namespace pik {

// kQuantWeights[N * N * c + N * y + x] is the relative weight of the (x, y)
// coefficient in component c. Higher weights correspond to finer quantization
// intervals and more bits spent in encoding.

namespace {
void GetQuantWeightsDCT2(const float dct2weights[3][6], double* weights) {
  for (size_t c = 0; c < 3; c++) {
    size_t start = c * 64;
    weights[start] = 0xBAD;
    weights[start + 1] = weights[start + 8] = dct2weights[c][0];
    weights[start + 9] = dct2weights[c][1];
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + y * 8 + x + 2] = dct2weights[c][2];
        weights[start + (y + 2) * 8 + x] = dct2weights[c][2];
      }
    }
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + (y + 2) * 8 + x + 2] = dct2weights[c][3];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + y * 8 + x + 4] = dct2weights[c][4];
        weights[start + (y + 4) * 8 + x] = dct2weights[c][4];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + (y + 4) * 8 + x + 4] = dct2weights[c][5];
      }
    }
  }
}

const double* GetQuantWeightsLines() {
  // The first value does not matter: it is the DC which is quantized elsewhere.
  static const double kPositionWeights[64] = {
      0,   100, 100, 100, 100, 100, 100, 5, 100, 100, 50, 20, 20, 10, 5, 5,
      100, 100, 50,  20,  20,  10,  5,   5, 100, 50,  50, 20, 20, 10, 5, 5,
      100, 20,  20,  20,  20,  10,  5,   5, 100, 10,  10, 10, 10, 10, 5, 5,
      100, 5,   5,   5,   5,   5,   5,   5, 5,   5,   5,  5,  5,  5,  5, 5,
  };
  static const double kChannelWeights[3] = {0.2, 0.5, 0.01};
  static const double kGlobal = 35.0;

  static double kQuantWeights[3 * 8 * 8] = {};

  for (size_t c = 0; c < 3; c++) {
    size_t start = c * 64;
    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        kQuantWeights[start + y * 8 + x] =
            kPositionWeights[y * 8 + x] * kChannelWeights[c] * kGlobal;
      }
    }
  }
  return kQuantWeights;
}

void GetQuantWeightsIdentity(const float idweights[3][3], double* weights) {
  for (size_t c = 0; c < 3; c++) {
    for (int i = 0; i < 64; i++) {
      weights[64 * c + i] = idweights[c][0];
    }
    weights[64 * c + 1] = idweights[c][1];
    weights[64 * c + 8] = idweights[c][1];
    weights[64 * c + 9] = idweights[c][2];
  }
}

// Computes quant weights for a SX*SY-sized transform, using num_bands
// eccentricity bands and num_ebands eccentricity bands. If print_mode is 1,
// prints the resulting matrix; if print_mode is 2, prints the matrix in a
// format suitable for a 3d plot with gnuplot.
template <size_t SX, size_t SY, size_t print_mode = 0>
Status GetQuantWeights(
    const float distance_bands[3][DctQuantWeightParams::kMaxDistanceBands],
    size_t num_bands,
    const float eccentricity_bands[3][DctQuantWeightParams::kMaxRadialBands],
    size_t num_ebands, double* out) {
  auto mult = [](double v) {
    if (v > 0) return 1 + v;
    return 1 / (1 - v);
  };

  auto interpolate = [](double pos, double max, double* array, size_t len) {
    double scaled_pos = pos * (len - 1) / max;
    size_t idx = scaled_pos;
    PIK_ASSERT(idx + 1 < len);
    double a = array[idx];
    double b = array[idx + 1];
    return a * pow(b / a, scaled_pos - idx);
  };

  for (size_t c = 0; c < 3; c++) {
    if (print_mode) {
      fprintf(stderr, "Channel %lu\n", c);
    }
    double bands[DctQuantWeightParams::kMaxDistanceBands] = {
        distance_bands[c][0]};
    for (size_t i = 1; i < num_bands; i++) {
      bands[i] = bands[i - 1] * mult(distance_bands[c][i]);
      if (bands[i] < 0) return PIK_FAILURE("Invalid distance bands");
    }
    double ebands[DctQuantWeightParams::kMaxRadialBands + 1] = {1.0};
    for (size_t i = 1; i <= num_ebands; i++) {
      ebands[i] = ebands[i - 1] * mult(eccentricity_bands[c][i - 1]);
      if (ebands[i] < 0) return PIK_FAILURE("Invalid eccentricity bands");
    }
    for (size_t y = 0; y < SY; y++) {
      for (size_t x = 0; x < SX; x++) {
        double dx = 1.0 * x / (SX - 1);
        double dy = 1.0 * y / (SY - 1);
        double distance = std::sqrt(dx * dx + dy * dy);
        double wd =
            interpolate(distance, std::sqrt(2) + 1e-6, bands, num_bands);
        double eccentricity =
            (x == 0 && y == 0) ? 0 : std::abs((double)dx - dy) / distance;
        double we =
            interpolate(eccentricity, 1.0 + 1e-6, ebands, num_ebands + 1);
        double weight = we * wd;

        if (print_mode == 1) {
          fprintf(stderr, "%15.12f, ", weight);
        }
        if (print_mode == 2) {
          fprintf(stderr, "%lu %lu %15.12f\n", x, y, weight);
        }
        out[c * SX * SY + y * SX + x] = weight;
      }
      if (print_mode) fprintf(stderr, "\n");
      if (print_mode == 1) fprintf(stderr, "\n");
    }
    if (print_mode) fprintf(stderr, "\n");
  }
  return true;
}

// TODO(veluca): use proper encoding for floats. If not, use integer
// encoding/decoding functions from byte_order.h. Also consider moving those
// fields to use the header machinery.
void EncodeUint(uint32_t v, std::string* s) {
  *s += (uint8_t)(v >> 24);
  *s += (uint8_t)(v >> 16);
  *s += (uint8_t)(v >> 8);
  *s += (uint8_t)v;
}

void EncodeFloat(float v, std::string* s) {
  static_assert(sizeof(float) == sizeof(uint32_t),
                "Float should be composed of 32 bits!");
  uint32_t tmp;
  memcpy(&tmp, &v, sizeof(float));
  EncodeUint(tmp, s);
}

uint32_t DecodeUint(BitReader* br) {
  br->FillBitBuffer();
  uint32_t v = br->ReadBits(8);
  v = (v << 8) | br->ReadBits(8);
  v = (v << 8) | br->ReadBits(8);
  v = (v << 8) | br->ReadBits(8);
  return v;
}

float DecodeFloat(BitReader* br) {
  uint32_t tmp = DecodeUint(br);
  float ret;
  memcpy(&ret, &tmp, sizeof(float));
  return ret;
}

void EncodeDctParams(const DctQuantWeightParams& params, std::string* s) {
  s += (uint8_t)params.num_distance_bands;
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params.num_distance_bands; i++) {
      EncodeFloat(params.distance_bands[c][i], s);
    }
  }
  s += (uint8_t)params.num_eccentricity_bands;
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params.num_eccentricity_bands; i++) {
      EncodeFloat(params.eccentricity_bands[c][i], s);
    }
  }
}

Status DecodeDctParams(BitReader* br, DctQuantWeightParams* params) {
  br->FillBitBuffer();
  if (params->num_distance_bands > DctQuantWeightParams::kMaxDistanceBands)
    return PIK_FAILURE("Too many distance bands");
  if (params->num_distance_bands == 0)
    return PIK_FAILURE("Too few distance bands");
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params->num_distance_bands; i++) {
      params->distance_bands[c][i] = DecodeFloat(br);
    }
  }
  br->FillBitBuffer();
  params->num_eccentricity_bands = br->ReadBits(8);
  if (params->num_eccentricity_bands > DctQuantWeightParams::kMaxRadialBands)
    return PIK_FAILURE("Too many eccentricity bands");
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params->num_eccentricity_bands; i++) {
      params->eccentricity_bands[c][i] = DecodeFloat(br);
    }
  }
  return true;
}

std::string Encode(const QuantEncoding& encoding) {
  std::string out(1, encoding.mode);
  switch (encoding.mode) {
    case QuantEncoding::kQuantModeLibrary: {
      out += encoding.predefined;
      break;
    }
    case QuantEncoding::kQuantModeID: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 3; i++) {
          EncodeFloat(encoding.idweights[c][i], &out);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 6; i++) {
          EncodeFloat(encoding.dct2weights[c][i], &out);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 2; i++) {
          EncodeFloat(encoding.dct4multipliers[c][i], &out);
        }
      }
      EncodeDctParams(encoding.dct_params, &out);
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      EncodeDctParams(encoding.dct_params, &out);
      break;
    }
    case QuantEncoding::kQuantModeRaw: {
      out += (uint8_t)encoding.block_dim;
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < encoding.block_dim * kBlockDim; y++) {
          for (size_t x = 0; x < encoding.block_dim * kBlockDim; x++) {
            if (x < encoding.block_dim && y < encoding.block_dim) continue;
            EncodeFloat(
                encoding.weights[c][y * encoding.block_dim * kBlockDim + x],
                &out);
          }
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeRawScaled: {
      out += (uint8_t)encoding.block_dim;
      for (size_t y = 0; y < encoding.block_dim * kBlockDim; y++) {
        for (size_t x = 0; x < encoding.block_dim * kBlockDim; x++) {
          if (x < encoding.block_dim && y < encoding.block_dim) continue;
          EncodeFloat(
              encoding.weights[0][y * encoding.block_dim * kBlockDim + x],
              &out);
        }
      }
      for (size_t c = 0; c < 3; c++) {
        EncodeFloat(encoding.scales[c], &out);
      }
      break;
    }
    case QuantEncoding::kQuantModeCopy: {
      out += encoding.source;
      break;
    }
  }
  return out;
}

Status Decode(BitReader* br, QuantEncoding* encoding, size_t required_size,
              size_t idx) {
  br->FillBitBuffer();
  int mode = br->ReadBits(8);
  switch (mode) {
    case QuantEncoding::kQuantModeLibrary: {
      br->FillBitBuffer();
      encoding->predefined = br->ReadBits(8);
      if (encoding->predefined >= kNumPredefinedTables) {
        return PIK_FAILURE("Invalid predefined table");
      }
      break;
    }
    case QuantEncoding::kQuantModeID: {
      if (required_size != 1) return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 3; i++) {
          encoding->idweights[c][i] = DecodeFloat(br);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      if (required_size != 1) return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 6; i++) {
          encoding->dct2weights[c][i] = DecodeFloat(br);
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      if (required_size != 1) return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 2; i++) {
          encoding->dct4multipliers[c][i] = DecodeFloat(br);
        }
      }
      PIK_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      PIK_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
      break;
    }
    case QuantEncoding::kQuantModeRaw: {
      br->FillBitBuffer();
      encoding->block_dim = br->ReadBits(8);
      if (required_size != encoding->block_dim)
        return PIK_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < encoding->block_dim * kBlockDim; y++) {
          for (size_t x = 0; x < encoding->block_dim * kBlockDim; x++) {
            // Override LLF values in the quantization table with invalid
            // values.
            if (x < encoding->block_dim && y < encoding->block_dim) {
              encoding->weights[c][y * encoding->block_dim * kBlockDim + x] =
                  0xBAD;
              continue;
            }
            encoding->weights[c][y * encoding->block_dim * kBlockDim + x] =
                DecodeFloat(br);
          }
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeRawScaled: {
      br->FillBitBuffer();
      encoding->block_dim = br->ReadBits(8);
      if (required_size != encoding->block_dim)
        return PIK_FAILURE("Invalid mode");
      for (size_t y = 0; y < encoding->block_dim * kBlockDim; y++) {
        for (size_t x = 0; x < encoding->block_dim * kBlockDim; x++) {
          // Override LLF values in the quantization table with invalid values.
          if (x < encoding->block_dim && y < encoding->block_dim) {
            encoding->weights[0][y * encoding->block_dim * kBlockDim + x] =
                0xBAD;
            continue;
          }
          encoding->weights[0][y * encoding->block_dim * kBlockDim + x] =
              DecodeFloat(br);
        }
      }
      for (size_t c = 0; c < 3; c++) {
        encoding->scales[c] = DecodeFloat(br);
      }
      break;
    }
    case QuantEncoding::kQuantModeCopy: {
      br->FillBitBuffer();
      encoding->source = br->ReadBits(8);
      if (encoding->source >= idx) {
        return PIK_FAILURE("Invalid source table");
      }
      break;
    }
    default:
      return PIK_FAILURE("Invalid quantization table encoding");
  }
  encoding->mode = QuantEncoding::Mode(mode);
  if (!br->Healthy()) return PIK_FAILURE("Failed reading quantization tables");
  return true;
}

Status ComputeQuantTable(const QuantEncoding& encoding, float* table,
                         size_t* offsets, QuantKind kind, size_t* pos) {
  double weights[3 * kMaxQuantTableSize];
  double numerators[kMaxQuantTableSize];
  decltype(&GetQuantWeights<8, 8>) get_dct_weights = nullptr;

  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  const float* idct4_scales = IDCTScales<N / 2>();
  const float* idct_scales = IDCTScales<N>();
  const float* idct16_scales = IDCTScales<2 * N>();
  const float* idct32_scales = IDCTScales<4 * N>();
  size_t num = 0;
  switch (kind) {
    case kQuantKindDCT8: {
      num = block_size;
      get_dct_weights = GetQuantWeights<8, 8>;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        const float idct_scale = idct_scales[x] * idct_scales[y] / num;
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindDCT16: {
      num = 4 * block_size;
      get_dct_weights = GetQuantWeights<16, 16>;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (2 * N);
        const size_t y = i / (2 * N);
        const float idct_scale = idct16_scales[x] * idct16_scales[y] / num;
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindDCT32: {
      num = 16 * block_size;
      get_dct_weights = GetQuantWeights<32, 32>;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (4 * N);
        const size_t y = i / (4 * N);
        const float idct_scale = idct32_scales[x] * idct32_scales[y] / num;
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindDCT4: {
      num = block_size;
      get_dct_weights = GetQuantWeights<4, 4>;
      for (size_t i = 0; i < N * N; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        float idct_scale =
            idct4_scales[x / 2] * idct4_scales[y / 2] / (N / 2 * N / 2);
        numerators[i] = idct_scale;
      }
      break;
    }
    case kQuantKindID:
    case kQuantKindDCT2:
    case kQuantKindLines: {
      get_dct_weights = GetQuantWeights<8, 8>;
      num = block_size;
      std::fill_n(numerators, block_size, 1.0);
      break;
    }
    case kNumQuantKinds: {
      PIK_ASSERT(false);
    }
  }
  PIK_ASSERT(get_dct_weights != nullptr);

  switch (encoding.mode) {
    case QuantEncoding::kQuantModeLibrary:
    case QuantEncoding::kQuantModeCopy: {
      // Library and copy quant encoding should get replaced by the actual
      // parameters by the caller.
      PIK_ASSERT(false);
      break;
    }
    case QuantEncoding::kQuantModeID: {
      PIK_ASSERT(num == block_size);
      GetQuantWeightsIdentity(encoding.idweights, weights);
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      PIK_ASSERT(num == block_size);
      GetQuantWeightsDCT2(encoding.dct2weights, weights);
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      PIK_ASSERT(num == block_size);
      double weights4x4[3 * 4 * 4];
      PIK_RETURN_IF_ERROR(get_dct_weights(
          encoding.dct_params.distance_bands,
          encoding.dct_params.num_distance_bands,
          encoding.dct_params.eccentricity_bands,
          encoding.dct_params.num_eccentricity_bands, weights4x4));
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < kBlockDim; y++) {
          for (size_t x = 0; x < kBlockDim; x++) {
            weights[c * num + y * kBlockDim + x] =
                weights4x4[c * 16 + (y / 2) * 4 + (x / 2)];
          }
        }
        weights[c * num + 1] /= encoding.dct4multipliers[c][0];
        weights[c * num + N] /= encoding.dct4multipliers[c][0];
        weights[c * num + N + 1] /= encoding.dct4multipliers[c][1];
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      PIK_RETURN_IF_ERROR(
          get_dct_weights(encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands,
                          encoding.dct_params.eccentricity_bands,
                          encoding.dct_params.num_eccentricity_bands, weights));
      break;
    }
    case QuantEncoding::kQuantModeRaw: {
      PIK_ASSERT(num == encoding.block_dim * encoding.block_dim * block_size);
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < num; i++) {
          weights[c * num + i] = encoding.weights[c][i];
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeRawScaled: {
      PIK_ASSERT(num == encoding.block_dim * encoding.block_dim * block_size);
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < num; i++) {
          weights[c * num + i] = encoding.weights[0][i] * encoding.scales[c];
        }
      }
      break;
    }
  }
  for (size_t c = 0; c < 3; c++) {
    offsets[kind * 3 + c] = *pos;
    for (size_t i = 0; i < num; i++) {
      double val = numerators[i] / weights[c * num + i];
      if (val > std::numeric_limits<float>::max() || val < 0) {
        return PIK_FAILURE("Invalid quantization table");
      }
      table[(*pos)++] = val;
    }
  }
  return true;
}
}  // namespace

// This definition is needed before C++17.
constexpr size_t DequantMatrices::required_size_[kNumQuantKinds];

std::string DequantMatrices::Encode(PikImageSizeInfo* info) const {
  PIK_ASSERT(encodings_.size() < std::numeric_limits<uint8_t>::max());
  uint8_t num_tables = encodings_.size();
  while (num_tables > 0 &&
         encodings_[num_tables - 1].mode == QuantEncoding::kQuantModeLibrary &&
         encodings_[num_tables - 1].predefined == 0) {
    num_tables--;
  }
  std::string out(1, num_tables);
  for (size_t i = 0; i < num_tables; i++) {
    out += pik::Encode(encodings_[i]);
  }
  if (info != nullptr) {
    info->total_size += out.size();
  }
  return out;
}

Status DequantMatrices::Decode(BitReader* br) {
  br->FillBitBuffer();
  size_t num_tables = br->ReadBits(8);
  encodings_.clear();
  size_t num_full_tables = DivCeil(num_tables, size_t(kNumQuantKinds));
  if (num_full_tables == 0) num_full_tables = 1;
  encodings_.resize(num_full_tables * kNumQuantKinds,
                    QuantEncoding::Library(0));
  for (size_t i = 0; i < num_tables; i++) {
    PIK_RETURN_IF_ERROR(
        pik::Decode(br, &encodings_[i], required_size_[i % kNumQuantKinds], i));
  }
  return DequantMatrices::Compute();
}

float V(double v) { return static_cast<float>(v); }

Status DequantMatrices::Compute() {
  size_t pos = 0;

  static_assert(kNumQuantKinds == 7,
                "Update this function when adding new quantization kinds.");
  static_assert(kNumPredefinedTables == 1,
                "Update this function when adding new quantization matrices to "
                "the library.");

  QuantEncoding library[kNumPredefinedTables][kNumQuantKinds];

  // DCT8
  {
    static const float distance_bands[3][6] = {{
                                                   V(6.7128322747011593),
                                                   V(-0.75596993600717899),
                                                   V(-0.47741990264036249),
                                                   V(-0.81596269409665323),
                                                   V(0.068767170571484654),
                                                   V(-20.887837229035178),
                                               },
                                               {
                                                   V(1.0308008496910044),
                                                   V(0.12563824546332958),
                                                   V(-0.98580000474151119),
                                                   V(-0.74783541528315123),
                                                   V(-0.18837957703830949),
                                                   V(-0.50540560621792985),
                                               },
                                               {
                                                   V(0.52922318082990072),
                                                   V(-6.2099138554436495),
                                                   V(2.4559360555622511),
                                                   V(-1.8645272975104017),
                                                   V(2.1626488944731781),
                                                   V(-20.514468231628619),
                                               }};

    static const float eccentricity_bands[3][3] = {
        {
            V(0.020372599856493687),
            V(0.0060672219272973112),
            V(-0.037634794641950318),
        },
        {
            V(0.19964780548254896),
            V(-0.20598244512425934),
            V(0.22606880802424917),
        },
        {
            V(-0.53357069165890425),
            V(0.067877070499761022),
            V(2.3529080139232321),
        },
    };
    library[0][kQuantKindDCT8] = QuantEncoding::DCT(
        DctQuantWeightParams(distance_bands, eccentricity_bands));
  }

  // Identity
  {
    static const float weights[3][3] = {
        {
            V(174.50360988711236),
            V(7098.4292418698387),
            V(4459.2881530953237),
        },
        {
            V(29.181414754407044),
            V(1462.9387613234978),
            V(1364.8889051351412),
        },
        {
            V(10.427519104606029),
            V(23.975682913740158),
            V(11.132318126587421),
        },
    };
    library[0][kQuantKindID] =
        QuantEncoding::Identity(weights[0], weights[1], weights[2]);
  }

  // DCT2
  {
    static const float weights[3][6] = {
        {
            V(3838.4633860359086),
            V(2711.45620096628),
            V(740.86588368521473),
            V(673.9663156327548),
            V(146.0409913884842),
            V(71.829450601171018),
        },
        {
            V(855.89982430974862),
            V(835.22486787836522),
            V(268.7887798267422),
            V(161.58150295707284),
            V(46.818625352324425),
            V(28.025832307111365),
        },
        {
            V(135.95933746046285),
            V(100.36113442694905),
            V(52.759147600958094),
            V(54.55000110144173),
            V(10.61194822539392),
            V(6.7321557070577027),
        },
    };
    library[0][kQuantKindDCT2] =
        QuantEncoding::DCT2(weights[0], weights[1], weights[2]);
  }

  // DCT4 (quant_kind 3)
  {
    static const float distance_bands[3][4] = {
        {
            V(20.464243458003235),
            V(-1.3216361675651374),
            V(-0.90068227414064506),
            V(-0.51692149442719293),
        },
        {
            V(3.4892753025959551),
            V(-0.3851659055605578),
            V(-1.6024424566582844),
            V(-0.090185175016963492),
        },
        {
            V(2.0543507462254667),
            V(-17.083007167897751),
            V(1.1553317008558754),
            V(-17.06851301189084),
        },
    };

    static const float eccentricity_bands[3][2] = {
        {
            V(-1.6540674814777321),
            V(1.4353603203078817),
        },
        {
            V(0.23246389755392743),
            V(0.11670410074064763),
        },
        {
            V(0.039676509798850998),
            V(1.7114284305197651),
        },
    };
    static const float muls[3][2] = {
        {
            V(0.47188805913083881),
            V(0.74665256923039514),
        },
        {
            V(0.27688273718512119),
            V(0.32787026106006584),
        },
        {
            V(0.94572969005995233),
            V(1.649348791638829),
        },
    };
    library[0][kQuantKindDCT4] = QuantEncoding::DCT4(
        DctQuantWeightParams(distance_bands, eccentricity_bands), muls[0],
        muls[1], muls[2]);
  }

  // DCT16
  {
    static const float distance_bands[3][6] = {{
                                                   V(2.8081053178832627),
                                                   V(-2.4300085829870786),
                                                   V(0.11683860865233302),
                                                   V(-0.48546810937737683),
                                                   V(-772.68999845881376),
                                                   V(-30.167218264433497),
                                               },
                                               {
                                                   V(0.61651518963555374),
                                                   V(-0.89670752611689697),
                                                   V(-1.4823203833923126),
                                                   V(-0.4392530120704895),
                                                   V(-0.96459916681512592),
                                                   V(-4.5043195385133448),
                                               },
                                               {
                                                   V(0.35315014395417571),
                                                   V(-6.1622959506013206),
                                                   V(1.3987478239168303),
                                                   V(-5.221619505420998),
                                                   V(-87.102308097158911),
                                                   V(-29.330248661246706),
                                               }};

    static const float eccentricity_bands[3][3] = {
        {
            V(-0.1082223243760141),
            V(0.16581730095161393),
            V(-0.22834397719738264),
        },
        {
            V(0.064907061033690178),
            V(-0.07809582529363121),
            V(-0.044761862879806769),
        },
        {
            V(-0.23977989838080313),
            V(-0.14631104822608662),
            V(0.026626451443453436),
        },
    };
    library[0][kQuantKindDCT16] = QuantEncoding::DCT(
        DctQuantWeightParams(distance_bands, eccentricity_bands));
  }

  // DCT32
  {
    static const float distance_bands[3][8] = {{
                                                   V(0.84716094396432662),
                                                   V(-2.4766455452218108),
                                                   V(0.2471181572547147),
                                                   V(0.57650543843415769),
                                                   V(-4.0833701828342583),
                                                   V(-28.279479541125081),
                                                   V(1.8036899065163079),
                                                   V(39.052449003220673),
                                               },
                                               {
                                                   V(0.17234631384979648),
                                                   V(-1.1404450629580913),
                                                   V(-0.69128963252295739),
                                                   V(-0.53270455087075774),
                                                   V(-0.46759485378919513),
                                                   V(-0.89356535322414299),
                                                   V(0.65008570941628885),
                                                   V(-0.66302446211939114),
                                               },
                                               {
                                                   V(0.22743363189568044),
                                                   V(-11.670472775652776),
                                                   V(-4.9179016084759626),
                                                   V(-5.4264719484417459),
                                                   V(-10.370646227045418),
                                                   V(1.9002093523030437),
                                                   V(-2.6705664701413623),
                                                   V(-20.889766266401665),
                                               }};

    static const float eccentricity_bands[3][4] = {
        {
            V(-0.77613778421797797),
            V(0.8972017714545496),
            V(-0.93436764214829893),
            V(0.18670848590931757),
        },
        {
            V(0.089533427641859925),
            V(0.08358828409098),
            V(-0.094110728686133543),
            V(-0.1286652050040859),
        },
        {
            V(1.0095255806548),
            V(-1.5336522088790263),
            V(-6.9680701189357501),
            V(1.3664229471277314),
        },
    };
    library[0][kQuantKindDCT32] = QuantEncoding::DCT(
        DctQuantWeightParams(distance_bands, eccentricity_bands));
  }

  // Diagonal lines
  {
    static const float kPositionWeights[64] = {
        0,   100, 100, 100, 100, 100, 100, 5, 100, 100, 50, 20, 20, 10, 5, 5,
        100, 100, 50,  20,  20,  10,  5,   5, 100, 50,  50, 20, 20, 10, 5, 5,
        100, 20,  20,  20,  20,  10,  5,   5, 100, 10,  10, 10, 10, 10, 5, 5,
        100, 5,   5,   5,   5,   5,   5,   5, 5,   5,   5,  5,  5,  5,  5, 5,
    };
    static const float kChannelWeights[3] = {7.0, 17.5, 0.35};
    library[0][kQuantKindLines] =
        QuantEncoding::RawScaled(1, kPositionWeights, kChannelWeights);
  }

  table_memory_ = AllocateArray(encodings_.size() / kNumQuantKinds *
                                TotalTableSize() * sizeof(float));
  table_ = reinterpret_cast<float*>(table_memory_.get());
  table_offsets_.resize(encodings_.size() * 3);

  auto encodings = encodings_;

  for (size_t table = 0; table < encodings.size(); table++) {
    while (encodings[table].mode == QuantEncoding::kQuantModeCopy) {
      encodings[table] = encodings[encodings[table].source];
    }
    if (encodings[table].mode == QuantEncoding::kQuantModeLibrary) {
      encodings[table] =
          library[encodings[table].predefined][table % kNumQuantKinds];
    }
    PIK_RETURN_IF_ERROR(
        ComputeQuantTable(encodings[table], table_, table_offsets_.data(),
                          (QuantKind)(table % kNumQuantKinds), &pos));
  }

  PIK_ASSERT(pos == encodings.size() / kNumQuantKinds * TotalTableSize());

  size_ = pos;
  if (need_inv_matrices_) {
    inv_table_memory_ = AllocateArray(pos * sizeof(float));
    inv_table_ = reinterpret_cast<float*>(inv_table_memory_.get());
    for (size_t i = 0; i < pos; i++) {
      inv_table_[i] = 1.0f / table_[i];
    }
  }
  return true;
}

void FindBestDequantMatrices(
    float butteraugli_target, float intensity_multiplier, const Image3F& opsin,
    const ImageF& initial_quant_field, DequantMatrices* dequant_matrices,
    ImageB* control_field, uint8_t table_map[kMaxQuantControlFieldValue][256]) {
  // TODO(veluca): heuristics for in-bitstream quant tables. Notice that this
  // function does *not* know the exact values of the quant field
  // post-FindBestQuantization.
  *dequant_matrices = DequantMatrices(/*need_inv_matrices=*/true);
  *control_field = ImageB(DivCeil(opsin.xsize(), kTileDim),
                          DivCeil(opsin.ysize(), kTileDim));
  ZeroFillImage(control_field);
  memset(table_map, 0, kMaxQuantControlFieldValue * 256);
}

bool DecodeDequantControlField(BitReader* PIK_RESTRICT br,
                               ImageB* PIK_RESTRICT dequant_cf) {
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(br)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  for (size_t y = 0; y < dequant_cf->ysize(); ++y) {
    uint8_t* PIK_RESTRICT row = dequant_cf->Row(y);
    for (size_t x = 0; x < dequant_cf->xsize(); ++x) {
      br->FillBitBuffer();
      row[x] = decoder.ReadSymbol(entropy, br);
    }
  }
  PIK_RETURN_IF_ERROR(br->JumpToByteBoundary());
  return true;
}

std::string EncodeDequantControlField(const ImageB& dequant_cf,
                                      PikImageSizeInfo* info) {
  const size_t max_out_size = dequant_cf.xsize() * dequant_cf.ysize() + 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  std::vector<uint32_t> histogram(256);
  for (int y = 0; y < dequant_cf.ysize(); ++y) {
    for (int x = 0; x < dequant_cf.xsize(); ++x) {
      ++histogram[dequant_cf.ConstRow(y)[x]];
    }
  }
  std::vector<uint8_t> bit_depths(256);
  std::vector<uint16_t> bit_codes(256);
  BuildAndStoreHuffmanTree(histogram.data(), histogram.size(),
                           bit_depths.data(), bit_codes.data(), &storage_ix,
                           storage);
  const size_t histo_bits = storage_ix;
  for (int y = 0; y < dequant_cf.ysize(); ++y) {
    const uint8_t* PIK_RESTRICT row = dequant_cf.ConstRow(y);
    for (int x = 0; x < dequant_cf.xsize(); ++x) {
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

namespace {
void ComputeDequantControlFieldMapMask(
    const ImageI& quant_field, const ImageB& dequant_cf,
    bool table_mask[kMaxQuantControlFieldValue][256]) {
  for (size_t y = 0; y < quant_field.ysize(); y++) {
    const int* PIK_RESTRICT row_qf = quant_field.ConstRow(y);
    const uint8_t* PIK_RESTRICT row_cf =
        dequant_cf.ConstRow(y / kTileDimInBlocks);
    for (size_t x = 0; x < quant_field.xsize(); x++) {
      table_mask[row_cf[x / kTileDimInBlocks]][row_qf[x] - 1] = true;
    }
  }
}

}  // namespace

std::string EncodeDequantControlFieldMap(
    const ImageI& quant_field, const ImageB& dequant_cf,
    const uint8_t table_map[kMaxQuantControlFieldValue][256],
    PikImageSizeInfo* info) {
  bool table_mask[kMaxQuantControlFieldValue][256] = {};
  ComputeDequantControlFieldMapMask(quant_field, dequant_cf, table_mask);
  const size_t max_out_size = kMaxQuantControlFieldValue * 256 + 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  std::vector<uint32_t> histogram(256);
  for (int y = 0; y < kMaxQuantControlFieldValue; ++y) {
    for (int x = 0; x < 256; ++x) {
      if (!table_mask[y][x]) continue;
      ++histogram[table_map[y][x]];
    }
  }
  std::vector<uint8_t> bit_depths(256);
  std::vector<uint16_t> bit_codes(256);
  BuildAndStoreHuffmanTree(histogram.data(), histogram.size(),
                           bit_depths.data(), bit_codes.data(), &storage_ix,
                           storage);
  const size_t histo_bits = storage_ix;
  for (int y = 0; y < kMaxQuantControlFieldValue; ++y) {
    for (int x = 0; x < 256; ++x) {
      if (!table_mask[y][x]) continue;
      WriteBits(bit_depths[table_map[y][x]], bit_codes[table_map[y][x]],
                &storage_ix, storage);
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

bool DecodeDequantControlFieldMap(
    BitReader* PIK_RESTRICT br, const ImageI& quant_field,
    const ImageB& dequant_cf,
    uint8_t table_map[kMaxQuantControlFieldValue][256]) {
  bool table_mask[kMaxQuantControlFieldValue][256] = {};
  memset(table_map, 0, kMaxQuantControlFieldValue * 256 * sizeof(uint8_t));
  ComputeDequantControlFieldMapMask(quant_field, dequant_cf, table_mask);
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(br)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  for (size_t y = 0; y < kMaxQuantControlFieldValue; ++y) {
    for (size_t x = 0; x < 256; ++x) {
      if (!table_mask[y][x]) continue;
      br->FillBitBuffer();
      table_map[y][x] = decoder.ReadSymbol(entropy, br);
    }
  }
  PIK_RETURN_IF_ERROR(br->JumpToByteBoundary());
  return true;
}

}  // namespace pik
