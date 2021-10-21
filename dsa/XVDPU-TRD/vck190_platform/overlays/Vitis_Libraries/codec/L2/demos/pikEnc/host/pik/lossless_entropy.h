// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_LOSSLESS_ENTROPY_H_
#define PIK_LOSSLESS_ENTROPY_H_

#include "pik/padded_bytes.h"

namespace pik {

size_t encodeVarInt(size_t value, uint8_t* output);

size_t decodeVarInt(const uint8_t* input, size_t inputSize, size_t* pos);

// TODO(janwas): output to PaddedBytes for compatibility with brotli.h.

// Output size can have special meaning, in each case you must encode the
// data differently yourself and EntropyDecode will not be able to decode it.
// If 0, then compression was not able to reduce size and you should output
// uncompressed.
// If 1, then the input data has exactly one byte repeated size times, and
// you must RLE compress it (encode the amount of times the one value repeats)
bool MaybeEntropyEncode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size);

// Does not know or return the compressed size, must be known from external
// source.
bool MaybeEntropyDecode(const uint8_t* data, size_t size, size_t out_capacity,
                        uint8_t* out, size_t* out_size);

}  // namespace pik

#endif  // PIK_LOSSLESS_ENTROPY_H_
