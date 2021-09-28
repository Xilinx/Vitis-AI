// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CODEC_H_
#define PIK_CODEC_H_

// Interface for encoding/decoding images and their metadata.

#include <stddef.h>
#include <string>
#include <vector>
#include "pik/color_management.h"
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/image_ops.h"
#include "pik/metadata.h"

namespace pik {

// Per-channel interval, used to convert between (full-range) external and
// (bounded or unbounded) temp values. See external_image.cc for the definitions
// of temp/external.
struct CodecInterval {
  CodecInterval() {}
  constexpr CodecInterval(float min, float max) : min(min), width(max - min) {}
  // Defaults for temp.
  float min = 0.0f;
  float width = 1.0f;
};

using CodecIntervals = std::array<CodecInterval, 4>;  // RGB[A] or Y[A]

// Shared (read-only, no need for thread_local) data. Prefer reusing a single
// instance to avoid regenerating the color profiles.
struct CodecContext {
  CodecContext();

  // Index with CodecInOut.IsGray().
  const std::array<ColorEncoding, 2> c_srgb;
  const std::array<ColorEncoding, 2> c_linear_srgb;
};

// Allows passing arbitrary metadata to decoders (required for PNM).
class DecoderHints {
 public:
  // key=color_space, value=Description(c/pp): specify the ColorEncoding of
  //   the pixels for decoding. Otherwise, if the codec did not obtain an ICC
  //   profile from the image, assume sRGB.
  //
  // Strings are taken from the command line, so avoid spaces for convenience.
  void Add(const std::string& key, const std::string& value) {
    kv_.emplace_back(key, value);
  }

  // Calls func(key, value) in order of Add.
  template <class Func>
  void Foreach(const Func& func) const {
    for (const KeyValue& kv : kv_) {
      func(kv.key, kv.value);
    }
  }

 private:
  // Splitting into key/value avoids parsing in each codec.
  struct KeyValue {
    KeyValue(const std::string& key, const std::string& value)
        : key(key), value(value) {}

    std::string key;
    std::string value;
  };

  std::vector<KeyValue> kv_;
};

// Codecs supported by CodecInOut::Encode.
enum class Codec : uint32_t {
  kUnknown,  // for CodecFromExtension
  kPNG,
  kPNM,
};

std::vector<Codec> Values(Codec);

// Lower case ASCII including dot, e.g. ".png".
std::string ExtensionFromCodec(Codec codec);
Codec CodecFromExtension(const std::string& extension);

// An image and all its metadata, plus functions to encode/decode to/from
// other image codecs. Also used as the input/output type of PIK.
class CodecInOut {
 public:
  // "codec_context" must remain valid throughout the lifetime of this instance.
  explicit CodecInOut(const CodecContext* codec_context)
      : context_(codec_context) {}
  const CodecContext* Context() const { return context_; }

  // Move-only (allows storing in std::vector).
  CodecInOut(CodecInOut&&) = default;
  CodecInOut& operator=(CodecInOut&&) = default;

  // -- SIZE

  const size_t xsize() const { return color().xsize(); }
  const size_t ysize() const { return color().ysize(); }
  void ShrinkTo(size_t xsize, size_t ysize) {
    color_.ShrinkTo(xsize, ysize);
    if (HasAlpha()) alpha_.ShrinkTo(xsize, ysize);
  }

  // -- COLOR

  // If c_current.IsGray(), all planes must be identical.
  void SetFromImage(Image3F&& color, const ColorEncoding& c_current);

  // Sets image data from 8-bit sRGB pixel array in bytes.
  // Amount of input bytes per pixel must be:
  // (is_gray ? 1 : 3) + (has_alpha ? 1 : 0)
  Status SetFromSRGB(size_t xsize, size_t ysize, bool is_gray, bool has_alpha,
                     const uint8_t* pixels, const uint8_t* end,
                     ThreadPool* pool = nullptr);

  // Sets image data from 16-bit sRGB data.
  // Amount of input uint16_t's per pixel must be:
  // (is_gray ? 1 : 3) + (has_alpha ? 1 : 0)
  Status SetFromSRGB(size_t xsize, size_t ysize, bool is_gray, bool has_alpha,
                     const uint16_t* pixels, const uint16_t* end,
                     ThreadPool* pool = nullptr);

  // Sets image data from sRGB pixel array in bytes.
  // This low-level function supports both 8-bit and 16-bit data in bytes to
  // provide efficient access to arbitrary byte order.
  // Amount of input bytes per pixel must be:
  // ((is_gray ? 1 : 3) + (has_alpha ? 1 : 0)) * (is_16bit ? 2 : 1)
  // The ordering of the channels is interleaved RGBA or gray+alpha in that
  // order.
  // The 16-bit byte order is given by big_endian, and this has no effect when
  // is_16bit is false.
  Status SetFromSRGB(size_t xsize, size_t ysize, bool is_gray, bool has_alpha,
                     bool is_16bit, bool big_endian,
                     const uint8_t* pixels, const uint8_t* end,
                     ThreadPool* pool = nullptr);

  // Decodes "bytes". Sets dec_c_original to c_current (for later encoding).
  // dec_hints may specify the "color_space" (otherwise, defaults to sRGB).
  Status SetFromBytes(const PaddedBytes& bytes, ThreadPool* pool = nullptr);

  // Reads from file and calls SetFromBytes.
  Status SetFromFile(const std::string& pathname, ThreadPool* pool = nullptr);

  const Image3F& color() const { return color_; }

  // Returns whether the color image has identical planes. Once established by
  // Set*, remains unchanged until a subsequent Set*.
  bool IsGray() const { return c_current_.IsGray(); }

  const ColorEncoding c_current() const { return c_current_; }
  bool IsSRGB() const {
    return c_current_.white_point == WhitePoint::kD65 &&
           c_current_.primaries == Primaries::kSRGB &&
           c_current_.transfer_function == TransferFunction::kSRGB;
  }
  bool IsLinearSRGB() const {
    return c_current_.white_point == WhitePoint::kD65 &&
           c_current_.primaries == Primaries::kSRGB &&
           IsLinear(c_current_.transfer_function);
  }

  // Transforms color to c_desired and sets c_current to c_desired. Alpha
  // remains unchanged.
  Status TransformTo(const ColorEncoding& c_desired,
                     ThreadPool* pool = nullptr);

  // Copies this:rect, converts to c_desired, and allocates+fills out.
  Status CopyTo(const Rect& rect, const ColorEncoding& c_desired, Image3B* out,
                ThreadPool* pool = nullptr) const;
  Status CopyTo(const Rect& rect, const ColorEncoding& c_desired, Image3U* out,
                ThreadPool* pool = nullptr) const;
  Status CopyTo(const Rect& rect, const ColorEncoding& c_desired, Image3F* out,
                ThreadPool* pool = nullptr) const;
  Status CopyToSRGB(const Rect& rect, Image3B* out,
                    ThreadPool* pool = nullptr) const;

  // TODO(janwas): remove, use Metadata field instead
  bool HasOriginalBitsPerSample() const { return has_dec_bits_per_sample_; }
  size_t original_bits_per_sample() const {
    PIK_ASSERT(HasOriginalBitsPerSample());
    return dec_bits_per_sample_;
  }
  void SetOriginalBitsPerSample(size_t bit_depth) {
    dec_bits_per_sample_ = bit_depth;
    has_dec_bits_per_sample_ = true;
  }

  // -- ALPHA

  bool HasAlpha() const { return alpha_.xsize() != 0; }
  // Zero if all pixels are transparent.
  size_t AlphaBits() const {
    PIK_ASSERT(HasAlpha());
    return alpha_bits_;
  }
  const ImageU& alpha() const {
    PIK_ASSERT(HasAlpha());
    return alpha_;
  }

  void SetAlpha(ImageU&& alpha, size_t alpha_bits) {
    PIK_CHECK(alpha_bits == 8 || alpha_bits == 16);
    alpha_bits_ = alpha_bits;
    alpha_ = std::move(alpha);
    PIK_CHECK(DivCeil(alpha_.xsize(), kBlockDim) ==
                  DivCeil(color_.xsize(), kBlockDim) &&
              DivCeil(alpha_.ysize(), kBlockDim) ==
                  DivCeil(color_.ysize(), kBlockDim));
  }

  // Called if all alpha values are opaque.
  void RemoveAlpha() {
    alpha_ = ImageU();
    PIK_ASSERT(!HasAlpha());
  }

  // -- ENCODER

  // Replaces "bytes" with an encoding of pixels transformed from c_current
  // color space to c_desired.
  Status Encode(const Codec codec, const ColorEncoding& c_desired,
                size_t bits_per_sample, PaddedBytes* bytes,
                ThreadPool* pool = nullptr) const;

  // Deduces codec, calls Encode and writes to file.
  Status EncodeToFile(const ColorEncoding& c_desired, size_t bits_per_sample,
                      const std::string& pathname,
                      ThreadPool* pool = nullptr) const;

  // -- ENCODER OUTPUT:

  // Size [bytes] of encoded bitstream after encoding / before decoding.
  mutable size_t enc_size;

  // Encoder-specific function of its bits_per_sample argument. Used to compute
  // error tolerance in round trips.
  mutable size_t enc_bits_per_sample;

  // Range of temp channels for rescaling instead of clipping. Not yet supported
  // by any Codec.
  mutable CodecIntervals enc_temp_intervals;  // unused

  // -- DECODER INPUT/OUTPUT:

  // Used to set c_current for codecs that lack color space metadata.
  DecoderHints dec_hints;

  // Color space/ICC profile from the original source.
  // Used to reconstruct the original image without additional user input.
  ColorEncoding dec_c_original;

  // -- SHARED:

  // Optional text/EXIF metadata to store into / retrieve from bitstreams.
  Metadata metadata;

 private:
  // Initialized by ctor:
  const CodecContext* context_;  // Not owned, must remain valid.

  // Initialized by Set*:
  Image3F color_;  // In c_current color space; all planes equal if IsGray().
  ColorEncoding c_current_;  // Encoding the values in color_ are defined in.

  // Initialized by SetAlpha; only queried if HasAlpha.
  size_t alpha_bits_;
  ImageU alpha_;  // Empty or same size as color_.

  // From the original source; may differ from sizeof(T) * kBitsPerBytes.
  // Used to reconstruct the original image without additional user input.
  size_t dec_bits_per_sample_;
  bool has_dec_bits_per_sample_ = false;
};

}  // namespace pik

#endif  // PIK_CODEC_H_
