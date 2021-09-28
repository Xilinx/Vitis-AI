// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_EXTERNAL_IMAGE_H_
#define PIK_EXTERNAL_IMAGE_H_

// Interleaved image for color transforms and Codec.

#include <stddef.h>
#include <stdint.h>

#include "pik/codec.h"
#include "pik/data_parallel.h"
#include "pik/status.h"

namespace pik {

// Packed (no row padding), interleaved (RGBRGB) u8/u16/f32.
class ExternalImage {
 public:
  // Copies from existing interleaved image. Called by decoders. "big_endian"
  // only matters for bits_per_sample > 8. "end" is the STL-style end of "bytes"
  // for range checks, or null if unknown.
  ExternalImage(size_t xsize, size_t ysize, const ColorEncoding& c_current,
                bool has_alpha, size_t bits_per_alpha,
                size_t bits_per_sample, bool big_endian,
                const uint8_t* bytes, const uint8_t* end);

  // Copies pixels from rect and converts from c_current to c_desired. Called by
  // encoders and CodecInOut::CopyTo. alpha is nullptr iff !has_alpha.
  // If temp_intervals != null, fills them such that CopyTo can rescale to that
  // range. Otherwise, clamps temp to [0, 1].
  ExternalImage(ThreadPool* pool, const Image3F& color, const Rect& rect,
                const ColorEncoding& c_current, const ColorEncoding& c_desired,
                bool has_alpha, const ImageU* alpha, size_t bits_per_alpha,
                size_t bits_per_sample, bool big_endian,
                CodecIntervals* temp_intervals);

  // Indicates whether the ctor succeeded; if not, do not use this instance.
  Status IsHealthy() const { return is_healthy_; }

  // Sets "io" to a newly allocated copy with c_current color space.
  // Uses temp_intervals for rescaling if not null (NOTE: temp_intervals is
  // given as if a range of [0.0f-1.0f] would be used, even though it uses
  // [0.0f-255.0f] internally, to match the same parameter given to the
  // color converting constructor).
  Status CopyTo(const CodecIntervals* temp_intervals, ThreadPool* pool,
                CodecInOut* io) const;

  // Packed, interleaved pixels, for passing to encoders.
  const PaddedBytes& Bytes() const { return bytes_; }

  size_t xsize() const { return xsize_; }
  size_t ysize() const { return ysize_; }
  const ColorEncoding& c_current() const { return c_current_; }
  bool IsGray() const { return c_current_.IsGray(); }
  bool HasAlpha() const { return channels_ == 2 || channels_ == 4; }
  size_t BitsPerAlpha() const { return bits_per_alpha_; }
  size_t BitsPerSample() const { return bits_per_sample_; }
  bool BigEndian() const { return big_endian_; }

  uint8_t* Row(size_t y) { return bytes_.data() + y * row_size_; }
  const uint8_t* ConstRow(size_t y) const {
    return bytes_.data() + y * row_size_;
  }

 private:
  ExternalImage(size_t xsize, size_t ysize, const ColorEncoding& c_current,
                bool has_alpha, size_t bits_per_alpha, size_t bits_per_sample,
                bool big_endian);

  size_t xsize_;
  size_t ysize_;
  ColorEncoding c_current_;
  size_t channels_;
  // Per alpha channel value
  size_t bits_per_alpha_;
  // Per color channel
  size_t bits_per_sample_;
  bool big_endian_;
  size_t row_size_;
  PaddedBytes bytes_;
  bool is_healthy_;
};

}  // namespace pik

#endif  // PIK_EXTERNAL_IMAGE_H_
