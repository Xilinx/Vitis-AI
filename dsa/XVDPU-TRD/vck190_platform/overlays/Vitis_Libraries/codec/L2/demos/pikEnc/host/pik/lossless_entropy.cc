// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/lossless_entropy.h"
#include "pik/padded_bytes.h"

#define PIK_ENTROPY_CODER_FSE 1  // tANS; smallest results for DC
#define PIK_ENTROPY_CODER_PIK 2  // rANS
// Potentially helpful for synthetic images but not DC
#define PIK_ENTROPY_CODER_BROTLI 3

#ifndef PIK_ENTROPY_CODER
#define PIK_ENTROPY_CODER PIK_ENTROPY_CODER_FSE
#endif

#if PIK_ENTROPY_CODER == PIK_ENTROPY_CODER_FSE
//#include "fse_wrapper.h"
#include "FiniteStateEntropy/lib/fse.h"
#elif PIK_ENTROPY_CODER == PIK_ENTROPY_CODER_PIK
#include "pik/entropy_coder.h"
#elif PIK_ENTROPY_CODER == PIK_ENTROPY_CODER_BROTLI
#include "pik/brotli.h"
#else
#error "Add include for entropy coder"
#endif

namespace pik {

size_t encodeVarInt(size_t value, uint8_t* output) {
  size_t outputSize = 0;
  // While more than 7 bits of data are left,
  // store 7 bits and set the next byte flag
  while (value > 127) {
    // |128: Set the next byte flag
    output[outputSize++] = ((uint8_t)(value & 127)) | 128;
    // Remove the seven bits we just wrote
    value >>= 7;
  }
  output[outputSize++] = ((uint8_t)value) & 127;
  return outputSize;
}

size_t decodeVarInt(const uint8_t* input, size_t inputSize, size_t* pos) {
  size_t i, ret = 0;
  for (i = 0; *pos + i < inputSize && i < 10; ++i) {
    ret |= uint64_t(input[*pos + i] & 127) << uint64_t(7 * i);
    // If the next-byte flag is not set, stop
    if ((input[*pos + i] & 128) == 0) break;
  }
  // TODO: Return a decoding error if i == 10.
  *pos += i + 1;
  return ret;
}

#if PIK_ENTROPY_CODER == PIK_ENTROPY_CODER_FSE

bool MaybeEntropyEncode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size) {
  size_t cs = FSE_compress2(out, out_capacity, data, size, 255,
                            /*FSE_MAX_TABLELOG=*/12);
  if (FSE_isError(cs)) {
    printf("FSE enc error: %s !!!\n", FSE_getErrorName(cs));
    return PIK_FAILURE("FSE enc error");
  }
  *out_size = cs;
  return true;
}

bool MaybeEntropyDecode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size) {
  size_t ds = FSE_decompress(out, out_capacity, data, size);
  if (FSE_isError(ds)) {
    printf("FSE dec error: %s !!!\n", FSE_getErrorName(ds));
    return PIK_FAILURE("FSE dec error");
  }
  *out_size = ds;
  return true;
}

#elif PIK_ENTROPY_CODER == PIK_ENTROPY_CODER_PIK

// Entropy encode with pik ANS
bool EntropyEncodePikANS(const uint8_t* data, size_t size,
                         std::vector<uint8_t>* result) {
  static const int kAlphabetSize = 256;
  static const int kContext = 0;

  std::vector<int> histogram(kAlphabetSize, 0);
  for (size_t i = 0; i < size; i++) {
    histogram[data[i]]++;
  }
  size_t cost_bound =
      1000 + 4 * size + 8 +
      ((size_t)ANSPopulationCost(histogram.data(), kAlphabetSize, size) + 7) /
          8;
  result->resize(cost_bound, 0);

  uint8_t* storage = result->data();
  size_t pos = 0;

  pos += encodeVarInt(size, storage + pos);

  std::vector<ANSEncodingData> encoding_codes(1);
  size_t bitpos = 0;
  encoding_codes[0].BuildAndStore(&histogram[0], histogram.size(), &bitpos,
                                  storage + pos);

  std::vector<uint8_t> dummy_context_map;
  dummy_context_map.push_back(0);  // only 1 histogram
  ANSSymbolWriter writer(encoding_codes, dummy_context_map, &bitpos,
                         storage + pos);
  for (size_t i = 0; i < size; i++) {
    writer.VisitSymbol(data[i], kContext);
  }
  writer.FlushToBitStream();
  pos += ((bitpos + 7) >> 3);
  result->resize(pos);

  return true;
}

// Entropy decode with pik ANS
bool EntropyDecodePikANS(const uint8_t* data, size_t size,
                         std::vector<uint8_t>* result) {
  static const int kContext = 0;
  size_t pos = 0;
  size_t num_symbols = decodeVarInt(data, size, &pos);
  if (pos >= size) {
    return PIK_FAILURE("lossless pik ANS decode failed");
  }
  // TODO(lode): instead take expected decoded size as function parameter
  if (num_symbols > 16777216) {
    // Avoid large allocations, we never expect this many symbols for
    // the limited group sizes.
    return PIK_FAILURE("lossless pik ANS decode too large");
  }

  BitReader br(data + pos, size - pos);
  ANSCode codes;
  if (!DecodeANSCodes(1, 256, &br, &codes)) {
    return PIK_FAILURE("lossless pik ANS decode failed");
  }

  result->resize(num_symbols);
  ANSSymbolReader reader(&codes);
  for (size_t i = 0; i < num_symbols; i++) {
    br.FillBitBuffer();
    int read_symbol = reader.ReadSymbol(kContext, &br);
    (*result)[i] = read_symbol;
  }
  if (!reader.CheckANSFinalState()) {
    return PIK_FAILURE("lossless pik ANS decode final state failed");
  }

  return true;
}

bool IsRLECompressible(const uint8_t* data, size_t size) {
  if (size < 4) return false;
  uint8_t first = data[0];
  for (size_t i = 1; i < size; i++) {
    if (data[i] != first) return false;
  }
  return true;
}

// TODO(lode): avoid the copying between std::vector and data.
// Entropy encode with pik ANS
bool MaybeEntropyEncode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size) {
  if (IsRLECompressible(data, size)) {
    *out_size = 1;  // Indicate the codec should use RLE instead,
    return true;
  }
  std::vector<uint8_t> result;
  if (!EntropyEncodePikANS(data, size, &result)) {
    return PIK_FAILURE("lossless entropy encoding failed");
  }
  if (result.size() > size) {
    *out_size = 0;  // Indicate the codec should use uncompressed mode instead.
    return true;
  }
  if (result.size() > out_capacity) {
    return PIK_FAILURE("lossless entropy encoding out of capacity");
  }
  memcpy(out, result.data(), result.size());
  *out_size = result.size();
  return true;
}

// Entropy decode with pik ANS
bool MaybeEntropyDecode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size) {
  std::vector<uint8_t> result;
  if (!EntropyDecodePikANS(data, size, &result)) {
    return PIK_FAILURE("lossless entropy decoding failed");
  }
  if (result.size() > out_capacity) {
    return PIK_FAILURE("lossless entropy encoding out of capacity");
  }
  memcpy(out, result.data(), result.size());
  *out_size = result.size();
  return true;
}

#elif PIK_ENTROPY_CODER == PIK_ENTROPY_CODER_BROTLI

bool MaybeEntropyEncode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size) {
  *out_size = 0;
  PIK_RETURN_IF_ERROR(BrotliCompress(11, data, size, out, out_size));
  if (*out_size > out_capacity) {
    return PIK_FAILURE("MaybeEntropyEncode exceeded buffer");
  }
  return true;
}

bool MaybeEntropyDecode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size) {
  size_t bytes_read = 0;
  PaddedBytes padded_out;
  PIK_RETURN_IF_ERROR(
      BrotliDecompress(data, size, out_capacity, &bytes_read, &padded_out));
  *out_size = padded_out.size();
  memcpy(out, padded_out.data(), padded_out.size());
  return true;
}

#else
#error "Implement all PIK_ENTROPY_CODER"
#endif

}  // namespace pik
