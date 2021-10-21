// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/brotli.h"

#include <string.h>  // memcpy
#include <memory>
#include "brotli/decode.h"
#include "brotli/encode.h"
#include "pik/status.h"

namespace pik {

Status BrotliCompress(int quality, const uint8_t* in, const size_t in_size,
                      uint8_t* PIK_RESTRICT out,
                      size_t* PIK_RESTRICT total_out_size) {
  std::unique_ptr<BrotliEncoderState, decltype(BrotliEncoderDestroyInstance)*>
      enc(BrotliEncoderCreateInstance(nullptr, nullptr, nullptr),
          BrotliEncoderDestroyInstance);
  if (!enc) return PIK_FAILURE("BrotliEncoderCreateInstance failed");

  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_QUALITY, quality);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGWIN, 24);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGBLOCK, 0);

  const size_t kBufferSize = 128 * 1024;
  PaddedBytes temp_buffer(kBufferSize);

  size_t avail_in = in_size;
  const uint8_t* next_in = in;

  size_t total_out = 0;

  while (1) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer.data();
    if (!BrotliEncoderCompressStream(enc.get(), BROTLI_OPERATION_FINISH,
                                     &avail_in, &next_in, &avail_out, &next_out,
                                     &total_out)) {
      return PIK_FAILURE("Brotli compression failed");
    }
    out_size = next_out - temp_buffer.data();
    memcpy(out + *total_out_size, temp_buffer.data(), out_size);
    *total_out_size += out_size;
    if (BrotliEncoderIsFinished(enc.get())) break;
  }

  return true;
}

Status BrotliCompress(int quality, const PaddedBytes& in,
                      PaddedBytes* PIK_RESTRICT out) {
  std::unique_ptr<BrotliEncoderState, decltype(BrotliEncoderDestroyInstance)*>
      enc(BrotliEncoderCreateInstance(nullptr, nullptr, nullptr),
          BrotliEncoderDestroyInstance);
  if (!enc) return PIK_FAILURE("BrotliEncoderCreateInstance failed");

  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_QUALITY, quality);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGWIN, 24);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGBLOCK, 0);

  const size_t kBufferSize = 128 * 1024;
  PaddedBytes temp_buffer(kBufferSize);

  size_t insize = in.size();
  size_t avail_in = insize;
  const uint8_t* next_in = in.data();

  size_t total_out = 0;

  while (1) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer.data();
    if (!BrotliEncoderCompressStream(enc.get(), BROTLI_OPERATION_FINISH,
                                     &avail_in, &next_in, &avail_out, &next_out,
                                     &total_out)) {
      return PIK_FAILURE("Brotli compression failed");
    }
    out_size = next_out - temp_buffer.data();
    out->resize(out->size() + out_size);
    memcpy(out->data() + out->size() - out_size, temp_buffer.data(), out_size);
    if (BrotliEncoderIsFinished(enc.get())) break;
  }

  return true;
}

Status BrotliDecompress(const uint8_t* in, size_t max_input_size,
                        size_t max_output_size, size_t* PIK_RESTRICT bytes_read,
                        PaddedBytes* PIK_RESTRICT out) {
  std::unique_ptr<BrotliDecoderState, decltype(BrotliDecoderDestroyInstance)*>
      s(BrotliDecoderCreateInstance(nullptr, nullptr, nullptr),
        BrotliDecoderDestroyInstance);
  if (!s) return PIK_FAILURE("BrotliDecoderCreateInstance failed");

  const size_t kBufferSize = 128 * 1024;
  PaddedBytes temp_buffer(kBufferSize);

  size_t avail_in = max_input_size;
  if (max_input_size == 0) return false;
  const uint8_t* next_in = in;
  BrotliDecoderResult code;

  while (1) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer.data();
    code = BrotliDecoderDecompressStream(s.get(), &avail_in, &next_in,
                                         &avail_out, &next_out, nullptr);
    out_size = next_out - temp_buffer.data();
    out->resize(out->size() + out_size);
    if (out->size() > max_output_size)
      return PIK_FAILURE("Brotli output too large");
    memcpy(out->data() + out->size() - out_size, temp_buffer.data(), out_size);
    if (code != BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) break;
  }
  if (code != BROTLI_DECODER_RESULT_SUCCESS)
    return PIK_FAILURE("Brotli decompression failed");
  *bytes_read += (max_input_size - avail_in);
  return true;
}

Status BrotliDecompress(const PaddedBytes& in, size_t max_output_size,
                        size_t* PIK_RESTRICT bytes_read,
                        PaddedBytes* PIK_RESTRICT out) {
  return BrotliDecompress(in.data(), in.size(), max_output_size, bytes_read,
                          out);
}

}  // namespace pik
