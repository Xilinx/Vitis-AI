// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/external_image.h"

#include <string.h>

#include "pik/byte_order.h"
#include "pik/cache_aligned.h"

namespace pik {
namespace {

#define PIK_EXT_VERBOSE 0

#if PIK_EXT_VERBOSE
// For printing RGB values at this X within each line.
constexpr size_t kX = 1;
#endif

// Encoding CodecInOut using other codecs requires format conversions to their
// "External" representation:
// IO -[1]-> Temp01 -[CMS]-> Temp01 -[2dt]-> External
// For External -> IO, we need only demux and rescale.
//
// "Temp01" and "Temp255" are interleaved and have 1 or 3 non-alpha channels.
// Alpha is included in External but not Temp because it is neither color-
// transformed nor included in Image3F.
// "IO" is Image3F (range [0, 255]) + ImageU alpha.
//
// "Temp01" is in range float [0, 1] as required by the CMS, but cannot
// losslessly represent 8-bit integer values [0, 255] due to floating point
// precision, which will reflect as a loss in Image3F which uses float range
// [0, 255] instead, which may cause effects on butteraugli score. Therefore,
// only use Temp01 if CMS transformation to different color space is required.
//
// "Temp255" is in range float [0, 255] and can losslessly represent 8-bit
// integer values [0, 255], but has floating point loss for 16-bit integer
// values [0, 65535]. The latter is not an issue however since Image3F uses
// float [0, 255] so has the same loss (so no butteraugli score effect), and
// the loss is gone when outputting to external integer again.
//
// Summary of formats:
//   Name   |   Bits  |    Max   | Channels |   Layout    |  Alpha
// ---------+---------+----------+----------+-------------+---------
// External | 8,16,32 | 2^Bits-1 |  1,2,3,4 | Interleaved | Included
//  Temp01  |    32   |     1    |    1,3   | Interleaved | Separate
// Temp255  |    32   |    255   |    1,3   | Interleaved | Separate
//    IO    |    32   |    255   |    3,4   |   Planar    |  ImageU

// Number of external channels including alpha.
struct Channels1 {
  static const char* Name() { return "1"; }
};
struct Channels2 {
  static const char* Name() { return "2"; }
};
struct Channels3 {
  static const char* Name() { return "3"; }
};
struct Channels4 {
  static const char* Name() { return "4"; }
};

// Step 1: interleaved <-> planar and rescale [0, 1] <-> [0, 255]
struct Interleave {
  static PIK_INLINE void Image3ToTemp01(Channels1, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* PIK_RESTRICT row_temp) {
    const float* PIK_RESTRICT row_image1 = rect.ConstPlaneRow(image, 1, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      row_temp[x] = row_image1[x] * (1.0f / 255);
    }
  }

  static PIK_INLINE void Image3ToTemp01(Channels3, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* PIK_RESTRICT row_temp) {
    const float* PIK_RESTRICT row_image0 = rect.ConstPlaneRow(image, 0, y);
    const float* PIK_RESTRICT row_image1 = rect.ConstPlaneRow(image, 1, y);
    const float* PIK_RESTRICT row_image2 = rect.ConstPlaneRow(image, 2, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      row_temp[3 * x + 0] = row_image0[x] * (1.0f / 255);
      row_temp[3 * x + 1] = row_image1[x] * (1.0f / 255);
      row_temp[3 * x + 2] = row_image2[x] * (1.0f / 255);
    }
  }

  // Same implementation for 2/4 because neither Image3 nor Temp have alpha.
  static PIK_INLINE void Image3ToTemp01(Channels2, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* PIK_RESTRICT row_temp) {
    Image3ToTemp01(Channels1(), y, image, rect, row_temp);
  }

  static PIK_INLINE void Image3ToTemp01(Channels4, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* PIK_RESTRICT row_temp) {
    Image3ToTemp01(Channels3(), y, image, rect, row_temp);
  }

  static PIK_INLINE void Temp255ToImage3(Channels1,
                                         const float* PIK_RESTRICT row_temp,
                                         size_t y,
                                         const Image3F* PIK_RESTRICT image) {
    const size_t xsize = image->xsize();
    float* PIK_RESTRICT row0 = const_cast<float*>(image->PlaneRow(0, y));
    for (size_t x = 0; x < xsize; ++x) {
      row0[x] = row_temp[x];
    }

    for (size_t c = 1; c < 3; ++c) {
      float* PIK_RESTRICT row = const_cast<float*>(image->PlaneRow(c, y));
      memcpy(row, row0, xsize * sizeof(float));
    }
  }

  static PIK_INLINE void Temp255ToImage3(Channels3,
                                         const float* PIK_RESTRICT row_temp,
                                         size_t y,
                                         const Image3F* PIK_RESTRICT image) {
    float* PIK_RESTRICT row_image0 = const_cast<float*>(image->PlaneRow(0, y));
    float* PIK_RESTRICT row_image1 = const_cast<float*>(image->PlaneRow(1, y));
    float* PIK_RESTRICT row_image2 = const_cast<float*>(image->PlaneRow(2, y));
    for (size_t x = 0; x < image->xsize(); ++x) {
      row_image0[x] = row_temp[3 * x + 0];
      row_image1[x] = row_temp[3 * x + 1];
      row_image2[x] = row_temp[3 * x + 2];
    }
  }

  static PIK_INLINE void Temp255ToImage3(Channels2,
                                         const float* PIK_RESTRICT row_temp,
                                         size_t y,
                                         const Image3F* PIK_RESTRICT image) {
    Temp255ToImage3(Channels1(), row_temp, y, image);
  }

  static PIK_INLINE void Temp255ToImage3(Channels4,
                                         const float* PIK_RESTRICT row_temp,
                                         size_t y,
                                         const Image3F* PIK_RESTRICT image) {
    Temp255ToImage3(Channels3(), row_temp, y, image);
  }

};

// Step 2t: type conversion

// Same naming convention as Image: B=u8, U=u16, F=f32. kSize enables generic
// functions with Type and Order template arguments.
struct TypeB {
  static const char* Name() { return "B"; }
  static constexpr size_t kSize = 1;
  static constexpr uint16_t kMaxAlpha = 0xFF;
};
struct TypeU {
  static const char* Name() { return "U"; }
  static constexpr size_t kSize = 2;
  static constexpr uint16_t kMaxAlpha = 0xFFFF;
};
struct TypeF {
  static const char* Name() { return "F"; }
  static constexpr size_t kSize = 4;
  static constexpr uint16_t kMaxAlpha = 0xFFFF;
};

// Load/stores float "sample" (gray/color) from/to u8/u16/float.
struct Sample {
  template <class Order>
  static PIK_INLINE float FromExternal(TypeB, const uint8_t* external) {
    return *external;
  }

  template <class Order>
  static PIK_INLINE float FromExternal(TypeU, const uint8_t* external) {
    return Load16(Order(), external);
  }

  template <class Order>
  static PIK_INLINE float FromExternal(TypeF, const uint8_t* external) {
    const int32_t bits = Load32(Order(), external);
    float sample;
    memcpy(&sample, &bits, 4);
    return sample;
  }

  template <class Order>
  static PIK_INLINE void ToExternal(TypeB, const float sample,
                                    uint8_t* external) {
    PIK_ASSERT(0 <= sample && sample < 256);
    // Don't need std::round since sample value is positive.
    *external = static_cast<int>(sample + 0.5f);
  }

  template <class Order>
  static PIK_INLINE void ToExternal(TypeU, const float sample,
                                    uint8_t* external) {
    PIK_ASSERT(0 <= sample && sample < 65536);
    // Don't need std::round since sample value is positive.
    Store16(Order(), static_cast<int>(sample + 0.5f), external);
  }

  template <class Order>
  static PIK_INLINE void ToExternal(TypeF, const float sample,
                                    uint8_t* external) {
    int32_t bits;
    memcpy(&bits, &sample, 4);
    Store32(Order(), bits, external);
  }
};

// Load/stores uint32_t (8/16-bit range) "alpha" from/to u8/u16. Lossless.
struct Alpha {
  // Per-thread alpha statistics.
  struct Stats {
    // Bitwise AND of all alpha values; used to detect all-opaque alpha.
    uint32_t and_bits = 0xFFFF;

    // Bitwise OR; used to detect out of bounds values (i.e. > 255 for 8-bit).
    uint32_t or_bits = 0;

    // Prevents false sharing.
    uint8_t pad[CacheAligned::kAlignment - sizeof(and_bits) - sizeof(or_bits)];
  };

  static PIK_INLINE uint32_t FromExternal(TypeB, OrderLE,
                                          const uint8_t* external) {
    return *external;
  }

  // Any larger type implies 16-bit alpha. NOTE: if TypeF, the alpha is smaller
  // than other external values (subsequent bytes are uninitialized/ignored).
  template <typename Type, class Order>
  static PIK_INLINE uint32_t FromExternal(Type, Order,
                                          const uint8_t* external) {
    const uint32_t alpha = Load16(Order(), external);
    return alpha;
  }

  static PIK_INLINE void ToExternal(TypeB, OrderLE, const uint32_t alpha,
                                    uint8_t* external) {
    PIK_ASSERT(alpha < 256);
    *external = alpha;
  }

  // Any larger type implies 16-bit alpha. NOTE: if TypeF, the alpha is smaller
  // than other external values (subsequent bytes are uninitialized/ignored).
  template <typename Type, class Order>
  static PIK_INLINE void ToExternal(Type, Order, const uint32_t alpha,
                                    uint8_t* external) {
    Store16(Order(), alpha, external);
  }
};

// Step 2d: demux external into separate (type-converted) color and alpha.
// Supports Temp01 and Temp255, the Cast decides this.
struct Demux {
  // 1 plane - copy all.
  template <class Type, class Order, class Cast>
  static PIK_INLINE void ExternalToTemp(Type type, Order order, Channels1,
                                        const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* PIK_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded =
          Sample::FromExternal<Order>(type, external + x * Type::kSize);
      row_temp[x] = cast.FromExternal(rounded, 0);
    }
  }
  template <class Type, class Order, class Cast>
  static PIK_INLINE void TempToExternal(Type type, Order order, Channels1,
                                        const size_t xsize,
                                        const float* PIK_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample = cast.FromTemp(row_temp[x], 0);
      Sample::ToExternal<Order>(type, sample, row_external + x * Type::kSize);
    }
  }

  // 2 planes - ignore alpha.
  template <class Type, class Order, class Cast>
  static PIK_INLINE void ExternalToTemp(Type type, Order order, Channels2,
                                        const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* PIK_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded = Sample::FromExternal<Order>(
          type, external + (2 * x + 0) * Type::kSize);
      row_temp[x] = cast.FromExternal(rounded, 0);
    }
  }
  template <class Type, class Order, class Cast>
  static PIK_INLINE void TempToExternal(Type type, Order order, Channels2,
                                        const size_t xsize,
                                        const float* PIK_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample = cast.FromTemp(row_temp[x], 0);
      Sample::ToExternal<Order>(type, sample,
                                row_external + (2 * x + 0) * Type::kSize);
    }
  }

  // 3 planes - copy all.
  template <class Type, class Order, class Cast>
  static PIK_INLINE void ExternalToTemp(Type type, Order order, Channels3,
                                        const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* PIK_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded0 = Sample::FromExternal<Order>(
          type, external + (3 * x + 0) * Type::kSize);
      const float rounded1 = Sample::FromExternal<Order>(
          type, external + (3 * x + 1) * Type::kSize);
      const float rounded2 = Sample::FromExternal<Order>(
          type, external + (3 * x + 2) * Type::kSize);
      row_temp[3 * x + 0] = cast.FromExternal(rounded0, 0);
      row_temp[3 * x + 1] = cast.FromExternal(rounded1, 1);
      row_temp[3 * x + 2] = cast.FromExternal(rounded2, 2);
    }
  }
  template <class Type, class Order, class Cast>
  static PIK_INLINE void TempToExternal(Type type, Order order, Channels3,
                                        const size_t xsize,
                                        const float* PIK_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample0 = cast.FromTemp(row_temp[3 * x + 0], 0);
      const float sample1 = cast.FromTemp(row_temp[3 * x + 1], 1);
      const float sample2 = cast.FromTemp(row_temp[3 * x + 2], 2);
      Sample::ToExternal<Order>(type, sample0,
                                row_external + (3 * x + 0) * Type::kSize);
      Sample::ToExternal<Order>(type, sample1,
                                row_external + (3 * x + 1) * Type::kSize);
      Sample::ToExternal<Order>(type, sample2,
                                row_external + (3 * x + 2) * Type::kSize);
    }
  }

  // 4 planes - ignore alpha.
  template <class Type, class Order, class Cast>
  static PIK_INLINE void ExternalToTemp(Type type, Order order, Channels4,
                                        const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* PIK_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded0 = Sample::FromExternal<Order>(
          type, external + (4 * x + 0) * Type::kSize);
      const float rounded1 = Sample::FromExternal<Order>(
          type, external + (4 * x + 1) * Type::kSize);
      const float rounded2 = Sample::FromExternal<Order>(
          type, external + (4 * x + 2) * Type::kSize);
      row_temp[3 * x + 0] = cast.FromExternal(rounded0, 0);
      row_temp[3 * x + 1] = cast.FromExternal(rounded1, 1);
      row_temp[3 * x + 2] = cast.FromExternal(rounded2, 2);
    }
  }
  template <class Type, class Order, class Cast>
  static PIK_INLINE void TempToExternal(Type type, Order order, Channels4,
                                        const size_t xsize,
                                        const float* PIK_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample0 = cast.FromTemp(row_temp[3 * x + 0], 0);
      const float sample1 = cast.FromTemp(row_temp[3 * x + 1], 1);
      const float sample2 = cast.FromTemp(row_temp[3 * x + 2], 2);
      Sample::ToExternal<Order>(type, sample0,
                                row_external + (4 * x + 0) * Type::kSize);
      Sample::ToExternal<Order>(type, sample1,
                                row_external + (4 * x + 1) * Type::kSize);
      Sample::ToExternal<Order>(type, sample2,
                                row_external + (4 * x + 2) * Type::kSize);
    }
  }

  // Gray only, no alpha.
  template <class Type, class Order>
  static PIK_INLINE void ExternalToAlpha(Type type, Order order, Channels1,
                                         const size_t xsize,
                                         const uint8_t* external,
                                         uint16_t* PIK_RESTRICT row_alpha,
                                         const size_t thread,
                                         std::vector<Alpha::Stats>* stats) {}
  template <class Type, class Order>
  static PIK_INLINE void AlphaToExternal(Type type, Order order, Channels1,
                                         const size_t xsize,
                                         const uint16_t* PIK_RESTRICT row_alpha,
                                         uint8_t* row_external) {}

  // Gray + alpha.
  template <class Type, class Order>
  static PIK_INLINE void ExternalToAlpha(Type type, Order order, Channels2,
                                         const size_t xsize,
                                         const uint8_t* external,
                                         uint16_t* PIK_RESTRICT row_alpha,
                                         const size_t thread,
                                         std::vector<Alpha::Stats>* stats) {
    if (row_alpha == nullptr) return;
    uint32_t and_bits = 0xFFFF;
    uint32_t or_bits = 0;
    for (size_t x = 0; x < xsize; ++x) {
      const uint32_t alpha = Alpha::FromExternal(
          type, order, external + (2 * x + 1) * Type::kSize);
      and_bits &= alpha;
      or_bits |= alpha;
      row_alpha[x] = alpha;
    }
    (*stats)[thread].and_bits &= and_bits;
    (*stats)[thread].or_bits |= or_bits;
  }
  template <class Type, class Order>
  static PIK_INLINE void AlphaToExternal(Type type, Order order, Channels2,
                                         const size_t xsize,
                                         const uint16_t* PIK_RESTRICT row_alpha,
                                         uint8_t* row_external) {
    if (row_alpha == nullptr) {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, type.kMaxAlpha,
                          row_external + (2 * x + 1) * Type::kSize);
      }
    } else {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, row_alpha[x],
                          row_external + (2 * x + 1) * Type::kSize);
      }
    }
  }

  // RGB only, no alpha.
  template <class Type, class Order>
  static PIK_INLINE void ExternalToAlpha(Type type, Order order, Channels3,
                                         const size_t xsize,
                                         const uint8_t* external,
                                         uint16_t* PIK_RESTRICT row_alpha,
                                         const size_t thread,
                                         std::vector<Alpha::Stats>* stats) {}
  template <class Type, class Order>
  static PIK_INLINE void AlphaToExternal(Type type, Order order, Channels3,
                                         const size_t xsize,
                                         const uint16_t* PIK_RESTRICT row_alpha,
                                         uint8_t* row_external) {}

  // RGBA.
  template <class Type, class Order>
  static PIK_INLINE void ExternalToAlpha(Type type, Order order, Channels4,
                                         const size_t xsize,
                                         const uint8_t* external,
                                         uint16_t* PIK_RESTRICT row_alpha,
                                         const size_t thread,
                                         std::vector<Alpha::Stats>* stats) {
    if (row_alpha == nullptr) return;
    uint32_t and_bits = 0xFFFF;
    uint32_t or_bits = 0;
    for (size_t x = 0; x < xsize; ++x) {
      const uint32_t alpha = Alpha::FromExternal(
          type, order, external + (4 * x + 3) * Type::kSize);
      and_bits &= alpha;
      or_bits |= alpha;
      row_alpha[x] = alpha;
    }
    (*stats)[thread].and_bits &= and_bits;
    (*stats)[thread].or_bits |= or_bits;
  }
  template <class Type, class Order>
  static PIK_INLINE void AlphaToExternal(Type type, Order order, Channels4,
                                         const size_t xsize,
                                         const uint16_t* PIK_RESTRICT row_alpha,
                                         uint8_t* row_external) {
    if (row_alpha == nullptr) {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, type.kMaxAlpha,
                          row_external + (4 * x + 3) * Type::kSize);
      }
    } else {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, row_alpha[x],
                          row_external + (4 * x + 3) * Type::kSize);
      }
    }
  }
};

// Used to select the Transformer::DoRow overload to call.
struct ToExternal1 {};  // first phase: store to temp and compute min/max.
struct ToExternal2 {};  // second phase: rescale temp to external.
struct ToExternal {};   // single-pass, only usable with CastClip.

// For ToExternal - assumes known/static extents of temp values.
struct ExtentsStatic {};

// For ToExternal1 - computes extents of temp values.
class ExtentsDynamic {
 public:
  ExtentsDynamic(const size_t xsize, const size_t ysize,
                 const size_t num_threads, const ColorEncoding& c_desired)
      : temp_intervals_(c_desired.Channels()) {
    // Store all temp pixels here, convert to external in a second phase after
    // Finalize computes ChannelIntervals from min_max_.
    temp_ = ImageF(xsize * temp_intervals_, ysize);

    min_max_.resize(num_threads);
  }

  float* PIK_RESTRICT RowTemp(const size_t y) { return temp_.Row(y); }

  // Row size is obtained from temp_. NOTE: clamps temp values to kMax.
  PIK_INLINE void Update(const size_t thread, float* PIK_RESTRICT row_temp) {
    // row_temp is interleaved - keep track of current channel.
    size_t c = 0;
    for (size_t i = 0; i < temp_.xsize(); ++i, ++c) {
      if (c == temp_intervals_) c = 0;
      if (row_temp[i] > min_max_[thread].max[c]) {
        if (row_temp[i] > kMax) row_temp[i] = kMax;
        min_max_[thread].max[c] = row_temp[i];
      }
      if (row_temp[i] < min_max_[thread].min[c]) {
        if (row_temp[i] < -kMax) row_temp[i] = -kMax;
        min_max_[thread].min[c] = row_temp[i];
      }
    }
  }

  void Finalize(CodecIntervals* temp_intervals) const {
    // Any other ChannelInterval remains default-initialized.
    for (size_t c = 0; c < temp_intervals_; ++c) {
      float min = min_max_[0].min[c];
      float max = min_max_[0].max[c];
      for (size_t i = 1; i < min_max_.size(); ++i) {
        min = std::min(min, min_max_[i].min[c]);
        max = std::max(max, min_max_[i].max[c]);
      }
      // Update ensured these are clamped.
      PIK_ASSERT(-kMax <= min && min <= max && max <= kMax);
      (*temp_intervals)[c] = CodecInterval(min, max);
    }
  }

 private:
  // Larger values are probably invalid, so clamp to preserve some precision.
  static constexpr float kMax = 1E10;

  struct MinMax {
    MinMax() {
      for (size_t c = 0; c < 4; ++c) {
        min[c] = kMax;
        max[c] = -kMax;
      }
    }

    float min[4];
    float max[4];
    // Prevents false sharing.
    uint8_t pad[CacheAligned::kAlignment - sizeof(min) - sizeof(max)];
  };

  const size_t temp_intervals_;
  ImageF temp_;
  std::vector<MinMax> min_max_;
};

// For ToExternal1, which updates ExtentsDynamic without casting.
struct CastUnused {};

// Returns range of valid values for all channel.
CodecInterval GetInterval(const size_t bits_per_sample) {
  if (bits_per_sample == 32) {
    // This ensures ConvertImage produces an image with the same [0, 255]
    // range as its input, but increases round trip error by ~2x vs [0, 1].
    return CodecInterval(0.0f, 255.0f);
  } else {
    const float max = (1U << bits_per_sample) - 1;
    return CodecInterval(0, max);
  }
}


// Lossless conversion between [0, 1] and [min, min+width]. Width is 1 or
// > 1 ("unbounded", useful for round trip testing). This is used to scale to
// the external type and back to the arbitrary interval.
class CastRescale01 {
 public:
  static const char* Name() { return "Rescale01"; }
  CastRescale01(const CodecIntervals& temp_intervals,
                const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_min_[c] = temp_intervals[c].min;
      temp_mul_[c] = ext_interval.width / temp_intervals[c].width;
      external_min_[c] = ext_interval.min;
      external_mul_[c] = temp_intervals[c].width / ext_interval.width;
    }
#if PIK_EXT_VERBOSE >= 2
    printf("CastRescale01 min %f width %f %f\n", temp_intervals[0].min,
           temp_intervals[0].width, ext_interval.width);
#endif
  }

  PIK_INLINE float FromExternal(const float external, const size_t c) const {
    return (external - external_min_[c]) * external_mul_[c] + temp_min_[c];
  }
  PIK_INLINE float FromTemp(const float temp, const size_t c) const {
    return (temp - temp_min_[c]) * temp_mul_[c] + external_min_[c];
  }

 private:
  float temp_min_[4];
  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};


// Lossless conversion between [0, 255] and [min, min+width]. Width is 255 or
// > 255 ("unbounded", useful for round trip testing). This is used to scale to
// the external type and back to the arbitrary interval.
// NOTE: this rescaler exists to make CopyTo match the convention of
// "temp_intervals" used by the color converting constructor. In the external to
// IO case without color conversion, one normally does not use this parameter.
class CastRescale255 {
 public:
  static const char* Name() { return "Rescale255"; }
  CastRescale255(const CodecIntervals& temp_intervals,
                 const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_min_[c] = 255.0f * temp_intervals[c].min;
      temp_mul_[c] =
          ext_interval.width / temp_intervals[c].width * (1.0f / 255);
      external_min_[c] = ext_interval.min * (1.0f / 255);
      external_mul_[c] = 255.0f * temp_intervals[c].width / ext_interval.width;
    }
#if PIK_EXT_VERBOSE >= 2
    printf("CastRescale255 min %f width %f %f\n", temp_intervals[0].min,
           temp_intervals[0].width, ext_interval.width);
#endif
  }

  PIK_INLINE float FromExternal(const float external, const size_t c) const {
    return (external - external_min_[c]) * external_mul_[c] + temp_min_[c];
  }
  PIK_INLINE float FromTemp(const float temp, const size_t c) const {
    return (temp - temp_min_[c]) * temp_mul_[c] + external_min_[c];
  }

 private:
  float temp_min_[4];
  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};

// Converts between [0, 1] and the external type's range. Lossy because values
// outside [0, 1] are clamped - this is necessary for codecs that are not able
// to store min/width metadata.
class CastClip01 {
 public:
  static const char* Name() { return "Clip01"; }
  CastClip01(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_mul_[c] = ext_interval.width;
      external_min_[c] = ext_interval.min;
      external_mul_[c] = 1.0f / ext_interval.width;
    }
#if PIK_EXT_VERBOSE >= 2
    printf("CastClip01 width %f\n", ext_interval.width);
#endif
  }

  PIK_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp01 = (external - external_min_[c]) * external_mul_[c];
    return temp01;
  }
  PIK_INLINE float FromTemp(const float temp, const size_t c) const {
    return Clamp01(temp) * temp_mul_[c] + external_min_[c];
  }

 private:
  static PIK_INLINE float Clamp01(const float temp) {
    return std::min(std::max(0.0f, temp), 1.0f);
  }

  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};

struct CastFloat {
  static const char* Name() { return "Float"; }
  CastFloat(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      PIK_CHECK(ext_interval.min == 0.0f);
      PIK_CHECK(ext_interval.width == 255.0f);
    }
#if PIK_EXT_VERBOSE >= 2
    printf("CastFloat\n");
#endif
  }

  PIK_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp01 = external * (1.0f / 255);
    return temp01;
  }
  PIK_INLINE float FromTemp(const float temp, const size_t c) const {
    return temp * 255.0f;
  }
};

// Converts between [0, 255] and the external type's range. Lossy because values
// outside [0, 255] are clamped - this is necessary for codecs that are not able
// to store min/width metadata.
class CastClip255 {
 public:
  static const char* Name() { return "Clip255"; }
  CastClip255(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_mul_[c] = ext_interval.width;
      external_min_[c] = ext_interval.min;
      external_mul_[c] = 255.0f / ext_interval.width;
    }
#if PIK_EXT_VERBOSE >= 2
    printf("CastClip255 width %f\n", ext_interval.width);
#endif
  }

  PIK_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp255 = (external - external_min_[c]) * external_mul_[c];
    return temp255;
  }
  PIK_INLINE float FromTemp(const float temp, const size_t c) const {
    return Clamp255(temp) * temp_mul_[c] + external_min_[c];
  }

 private:
  static PIK_INLINE float Clamp255(const float temp) {
    return std::min(std::max(0.0f, temp), 255.0f);
  }

  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};

struct CastFloat01 {
  static const char* Name() { return "Float01"; }
  CastFloat01(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      PIK_CHECK(ext_interval.min == 0.0f);
      PIK_CHECK(ext_interval.width == 255.0f);
    }
#if PIK_EXT_VERBOSE >= 2
    printf("CastFloat01\n");
#endif
  }

  PIK_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp01 = external * (1.0f / 255);
    return temp01;
  }
  PIK_INLINE float FromTemp(const float temp, const size_t c) const {
    return temp * 255.0f;
  }
};

// No-op
struct CastFloat255 {
  static const char* Name() { return "Float255"; }
  CastFloat255(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      PIK_CHECK(ext_interval.min == 0.0f);
      PIK_CHECK(ext_interval.width == 255.0f);
    }
#if PIK_EXT_VERBOSE >= 2
    printf("CastFloat255\n");
#endif
  }

  PIK_INLINE float FromExternal(const float external, const size_t c) const {
    return external;
  }
  PIK_INLINE float FromTemp(const float temp, const size_t c) const {
    return temp;
  }
};

// Multithreaded color space transform from IO to ExternalImage.
class Transformer {
 public:
  Transformer(ThreadPool* pool, const Image3F& color, const Rect& rect,
              const bool has_alpha, const ImageU* alpha,
              ExternalImage* external)
      : pool_(pool),
        color_(color),
        rect_(rect),
        alpha_(alpha),
        external_(external),
        want_alpha_(has_alpha && external->HasAlpha()) {
    PIK_ASSERT(rect.IsInside(color));
    PIK_ASSERT(SameSize(rect, *external));
  }

  // Can fail => separate from ctor.
  Status Init(const ColorEncoding& c_src, const ColorEncoding& c_dst) {
#if PIK_EXT_VERBOSE >= 1
    printf("%s->%s\n", Description(c_src).c_str(), Description(c_dst).c_str());
#endif

    return transform_.Init(c_src, c_dst, rect_.xsize(), NumThreads(pool_));
  }

  // Converts in the specified direction (To*).
  template <class To, class Extent, class Cast>
  Status Run(Extent* extents, const Cast& cast) {
    const size_t bytes = DivCeil(external_->BitsPerSample(), kBitsPerByte);
    const bool big_endian = external_->BigEndian();
    if (bytes == 1) {
      DispatchType<To, TypeB, OrderLE>(extents, cast);
    } else if (bytes == 2 && big_endian) {
      DispatchType<To, TypeU, OrderBE>(extents, cast);
    } else if (bytes == 2) {
      DispatchType<To, TypeU, OrderLE>(extents, cast);
    } else if (bytes == 4 && big_endian) {
      DispatchType<To, TypeF, OrderBE>(extents, cast);
    } else if (bytes == 4) {
      DispatchType<To, TypeF, OrderLE>(extents, cast);
    } else {
      return PIK_FAILURE("Unsupported BitsPerSample");
    }
    return true;
  }

 private:
  // First pass: only needed for ExtentsDynamic/CastUnused.
  template <class Type, class Order, class Channels>
  PIK_INLINE void DoRow(ToExternal1, ExtentsDynamic* extents, const CastUnused,
                        const size_t y, const size_t thread) {
    float* PIK_RESTRICT row_temp = extents->RowTemp(y);

    Interleave::Image3ToTemp01(Channels(), y, color_, rect_, row_temp);

#if PIK_EXT_VERBOSE
    const float in0 = row_temp[3 * kX + 0], in1 = row_temp[3 * kX + 1];
    const float in2 = row_temp[3 * kX + 2];
#endif

    transform_.Run(thread, row_temp, row_temp);

#if PIK_EXT_VERBOSE
    printf("ToExt1: in %.4f %.4f %.4f; xform %.4f %.4f %.4f\n", in0, in1, in2,
           row_temp[3 * kX + 0], row_temp[3 * kX + 1], row_temp[3 * kX + 2]);
#endif

    extents->Update(thread, row_temp);
  }

  // Second pass: only needed for ExtentsDynamic/CastRescale.
  template <class Type, class Order, class Channels>
  PIK_INLINE void DoRow(ToExternal2, ExtentsDynamic* extents,
                        const CastRescale01& cast, const size_t y,
                        const size_t thread) {
    const float* PIK_RESTRICT row_temp = extents->RowTemp(y);
    uint8_t* PIK_RESTRICT row_external = external_->Row(y);
    Demux::TempToExternal(Type(), Order(), Channels(), rect_.xsize(), row_temp,
                          cast, row_external);

#if PIK_EXT_VERBOSE
    printf("ToExt2: ext %3d %3d %3d\n", row_external[3 * kX + 0],
           row_external[3 * kX + 1], row_external[3 * kX + 2]);
#endif

    const uint16_t* PIK_RESTRICT row_alpha =
        want_alpha_ ? alpha_->ConstRow(y) : nullptr;
    Demux::AlphaToExternal(Type(), Order(), Channels(), rect_.xsize(),
                           row_alpha, row_external);
  }

  // Single-pass: only works for ExtentsStatic.
  template <class Type, class Order, class Channels, class Cast>
  PIK_INLINE void DoRow(ToExternal, ExtentsStatic*, const Cast& cast,
                        const size_t y, const size_t thread) {
    float* PIK_RESTRICT row_temp = transform_.BufDst(thread);
    Interleave::Image3ToTemp01(Channels(), y, color_, rect_, row_temp);

#if PIK_EXT_VERBOSE
    // Save inputs for printing before in-place transform overwrites them.
    const float in0 = row_temp[3 * kX + 0];
    const float in1 = row_temp[3 * kX + 1];
    const float in2 = row_temp[3 * kX + 2];
#endif
    transform_.Run(thread, row_temp, row_temp);

    uint8_t* PIK_RESTRICT row_external = external_->Row(y);
    Demux::TempToExternal(Type(), Order(), Channels(), rect_.xsize(), row_temp,
                          cast, row_external);

#if PIK_EXT_VERBOSE
    const float tmp0 = row_temp[3 * kX + 0];
    const float tmp1 = row_temp[3 * kX + 1];
    const float tmp2 = row_temp[3 * kX + 2];
    // Convert back so we can print the external values
    Demux::ExternalToTemp(Type(), Order(), Channels(), rect_.xsize(),
                          row_external, cast, row_temp);
    printf("ToExt(%s%s %s): tmp %.4f %.4f %.4f|%.4f %.4f %.4f|%.4f %.4f %.4f\n",
           Channels::Name(), Type::Name(), Cast::Name(), in0, in1, in2, tmp0,
           tmp1, tmp2, row_temp[3 * kX + 0], row_temp[3 * kX + 1],
           row_temp[3 * kX + 2]);
#endif

    const uint16_t* PIK_RESTRICT row_alpha =
        want_alpha_ ? alpha_->ConstRow(y) : nullptr;
    Demux::AlphaToExternal(Type(), Order(), Channels(), rect_.xsize(),
                           row_alpha, row_external);
  }

  // Closure callable by ThreadPool.
  template <class To, class Type, class Order, class Channels, class Extent,
            class Cast>
  class Bind {
   public:
    explicit Bind(Transformer* converter, Extent* extents, const Cast& cast)
        : xform_(converter), extents_(extents), cast_(cast) {}

    PIK_INLINE void operator()(const int task, const int thread) const {
      xform_->DoRow<Type, Order, Channels>(To(), extents_, cast_, task, thread);
    }

   private:
    Transformer* xform_;  // not owned
    Extent* extents_;     // not owned
    const Cast cast_;
  };

  template <class To, class Type, class Order, class Channels, class Extent,
            class Cast>
  void DoRows(Extent* extents, const Cast& cast) {
    RunOnPool(
        pool_, 0, rect_.ysize(),
        Bind<To, Type, Order, Channels, Extent, Cast>(this, extents, cast),
        "ExtImg xform");
  }

  // Calls the instantiation with the matching Type and Order.
  template <class To, class Type, class Order, class Extent, class Cast>
  void DispatchType(Extent* extents, const Cast& cast) {
    if (external_->IsGray()) {
      if (external_->HasAlpha()) {
        DoRows<To, Type, Order, Channels2>(extents, cast);
      } else {
        DoRows<To, Type, Order, Channels1>(extents, cast);
      }
    } else {
      if (external_->HasAlpha()) {
        DoRows<To, Type, Order, Channels4>(extents, cast);
      } else {
        DoRows<To, Type, Order, Channels3>(extents, cast);
      }
    }
  }

  ThreadPool* pool_;  // not owned
  const Image3F& color_;
  const Rect rect_;          // whence in color_ to copy, and output size.
  const ImageU* alpha_;      // not owned
  ExternalImage* external_;  // not owned

  bool want_alpha_;

  ColorSpaceTransform transform_;
};

// Multithreaded deinterleaving/conversion from ExternalImage to Image3.
class Converter {
 public:
  Converter(ThreadPool* pool, const ExternalImage& external)
      : pool_(pool),
        external_(&external),
        xsize_(external.xsize()),
        ysize_(external.ysize()),
        color_(xsize_, ysize_) {
    const size_t num_threads = NumThreads(pool);
    temp_buf_ = ImageF(xsize_ * external.c_current().Channels(), num_threads);

    if (external_->HasAlpha()) {
      alpha_ = ImageU(xsize_, ysize_);
      bits_per_alpha_ = external_->BitsPerAlpha();
      alpha_stats_.resize(num_threads);
    }
  }

  template <class Cast>
  Status Run(const Cast& cast) {
    const size_t bytes = DivCeil(external_->BitsPerSample(), kBitsPerByte);
    const bool big_endian = external_->BigEndian();
    if (bytes == 1) {
      DispatchType<TypeB, OrderLE>(cast);
    } else if (bytes == 2 && big_endian) {
      DispatchType<TypeU, OrderBE>(cast);
    } else if (bytes == 2) {
      DispatchType<TypeU, OrderLE>(cast);
    } else if (bytes == 4 && big_endian) {
      DispatchType<TypeF, OrderBE>(cast);
    } else if (bytes == 4) {
      DispatchType<TypeF, OrderLE>(cast);
    } else {
      return PIK_FAILURE("Unsupported BitsPerSample");
    }
    return true;
  }

  Status MoveTo(CodecInOut* io) {
    io->SetFromImage(std::move(color_), external_->c_current());

    // Don't have alpha; during TransformTo, don't remove existing alpha.
    if (alpha_stats_.empty()) return true;

    const size_t max_alpha = (1 << bits_per_alpha_) - 1;

    // Reduce per-thread statistics.
    uint32_t and_bits = alpha_stats_[0].and_bits;
    uint32_t or_bits = alpha_stats_[0].or_bits;
    for (size_t i = 1; i < alpha_stats_.size(); ++i) {
      and_bits &= alpha_stats_[i].and_bits;
      or_bits |= alpha_stats_[i].or_bits;
    }

    if (or_bits > max_alpha) {
      return PIK_FAILURE("Alpha out of range");
    }

    // Keep alpha if at least one value is (semi)transparent.
    if (and_bits != max_alpha) {
      io->SetAlpha(std::move(alpha_), bits_per_alpha_);
    } else {
      io->RemoveAlpha();
    }
    return true;
  }

 private:
  template <class Type, class Order, class Channels, class Cast>
  PIK_INLINE void DoRow(const Cast& cast, const size_t y, const size_t thread) {
    const uint8_t* PIK_RESTRICT row_external = external_->ConstRow(y);

    if (!alpha_stats_.empty()) {
      // No-op if Channels1/3.
      Demux::ExternalToAlpha(Type(), Order(), Channels(), xsize_, row_external,
                             alpha_.Row(y), thread, &alpha_stats_);
    }

    float* PIK_RESTRICT row_temp = temp_buf_.Row(thread);
    Demux::ExternalToTemp(Type(), Order(), Channels(), xsize_, row_external,
                          cast, row_temp);

#if PIK_EXT_VERBOSE
    printf("ToIO(%s%s %s): ext %3d %3d %3d  tmp %.4f %.4f %.4f\n",
           Channels::Name(), Type::Name(), Cast::Name(),
           row_external[3 * kX + 0], row_external[3 * kX + 1],
           row_external[3 * kX + 2], row_temp[3 * kX + 0], row_temp[3 * kX + 1],
           row_temp[3 * kX + 2]);
#endif

    Interleave::Temp255ToImage3(Channels(), row_temp, y, &color_);
  }

  // Closure callable by ThreadPool.
  template <class Type, class Order, class Channels, class Cast>
  class Bind {
   public:
    explicit Bind(Converter* converter, const Cast& cast)
        : converter_(converter), cast_(cast) {}

    PIK_INLINE void operator()(const int task, const int thread) const {
      converter_->DoRow<Type, Order, Channels>(cast_, task, thread);
    }

   private:
    Converter* converter_;  // not owned
    const Cast cast_;
  };

  template <class Type, class Order, class Channels, class Cast>
  void DoRows(const Cast& cast) {
    RunOnPool(pool_, 0, ysize_, Bind<Type, Order, Channels, Cast>(this, cast),
              "ExtImg cvt");
  }

  // Calls the instantiation with the matching Type and Order.
  template <class Type, class Order, class Cast>
  void DispatchType(const Cast& cast) {
    if (external_->IsGray()) {
      if (external_->HasAlpha()) {
        DoRows<Type, Order, Channels2>(cast);
      } else {
        DoRows<Type, Order, Channels1>(cast);
      }
    } else {
      if (external_->HasAlpha()) {
        DoRows<Type, Order, Channels4>(cast);
      } else {
        DoRows<Type, Order, Channels3>(cast);
      }
    }
  }

  ThreadPool* pool_;               // not owned
  const ExternalImage* external_;  // not owned
  size_t xsize_;
  size_t ysize_;
  Image3F color_;

  ImageF temp_buf_;

  // Only initialized if external_->HasAlpha() && want_alpha:
  std::vector<Alpha::Stats> alpha_stats_;
  ImageU alpha_;
  size_t bits_per_alpha_;
};

}  // namespace

ExternalImage::ExternalImage(const size_t xsize, const size_t ysize,
                             const ColorEncoding& c_current,
                             const bool has_alpha, const size_t bits_per_alpha,
                             const size_t bits_per_sample,
                             const bool big_endian)
    : xsize_(xsize),
      ysize_(ysize),
      c_current_(c_current),
      channels_(c_current.Channels() + has_alpha),
      bits_per_alpha_(bits_per_alpha),
      bits_per_sample_(bits_per_sample),
      big_endian_(big_endian),
      row_size_(xsize * channels_ * DivCeil(bits_per_sample, kBitsPerByte)) {
  PIK_ASSERT(1 <= channels_ && channels_ <= 4);
  PIK_ASSERT(1 <= bits_per_sample && bits_per_sample <= 32);
  if (has_alpha) PIK_ASSERT(1 <= bits_per_alpha && bits_per_alpha <= 32);
  bytes_.resize(ysize_ * row_size_);
  is_healthy_ = !bytes_.empty();
}

ExternalImage::ExternalImage(const size_t xsize, const size_t ysize,
                             const ColorEncoding& c_current,
                             const bool has_alpha, const size_t bits_per_alpha,
                             const size_t bits_per_sample,
                             const bool big_endian, const uint8_t* bytes,
                             const uint8_t* end)
    : ExternalImage(xsize, ysize, c_current, has_alpha, bits_per_alpha,
                    bits_per_sample, big_endian) {
  if (is_healthy_) {
    if (end != nullptr) PIK_CHECK(bytes + ysize * row_size_ <= end);
    memcpy(bytes_.data(), bytes, bytes_.size());
  }
}

ExternalImage::ExternalImage(ThreadPool* pool, const Image3F& color,
                             const Rect& rect, const ColorEncoding& c_current,
                             const ColorEncoding& c_desired,
                             const bool has_alpha, const ImageU* alpha,
                             size_t bits_per_alpha, size_t bits_per_sample,
                             bool big_endian,
                             CodecIntervals* temp_intervals)
    : ExternalImage(rect.xsize(), rect.ysize(), c_desired, has_alpha,
                    bits_per_alpha, bits_per_sample, big_endian) {
  if (!is_healthy_) return;
  Transformer transformer(pool, color, rect, has_alpha, alpha, this);
  if (!transformer.Init(c_current, c_desired)) {
    is_healthy_ = false;
    return;
  }

  const CodecInterval ext_interval = GetInterval(bits_per_sample);

  if (bits_per_sample == 32) {
    ExtentsStatic extents;
    const CastFloat01 cast(ext_interval);  // only multiply by const
    is_healthy_ = transformer.Run<ToExternal>(&extents, cast);
  } else if (temp_intervals != nullptr) {
    // Store temp to separate image and obtain per-channel intervals.
    ExtentsDynamic extents(xsize_, ysize_, NumThreads(pool), c_desired);
    const CastUnused unused;
    is_healthy_ = transformer.Run<ToExternal1>(&extents, unused);
    if (!is_healthy_) return;
    extents.Finalize(temp_intervals);

    // Rescale based on temp_intervals.
    const CastRescale01 cast(*temp_intervals, ext_interval);
    is_healthy_ = transformer.Run<ToExternal2>(&extents, cast);
  } else {
    ExtentsStatic extents;
    const CastClip01 cast(ext_interval);  // clip
    is_healthy_ = transformer.Run<ToExternal>(&extents, cast);
  }
}

Status ExternalImage::CopyTo(const CodecIntervals* temp_intervals,
                             ThreadPool* pool, CodecInOut* io) const {
  PIK_ASSERT(IsHealthy());  // Caller should have checked beforehand.

  Converter converter(pool, *this);

  const CodecInterval ext_interval = GetInterval(bits_per_sample_);

  if (bits_per_sample_ == 32) {
    const CastFloat255 cast(ext_interval);
    PIK_RETURN_IF_ERROR(converter.Run(cast));
  } else if (temp_intervals != nullptr) {
    const CastRescale255 cast(*temp_intervals, ext_interval);
    PIK_RETURN_IF_ERROR(converter.Run(cast));
  } else {
    const CastClip255 cast(ext_interval);
    PIK_RETURN_IF_ERROR(converter.Run(cast));
  }

  return converter.MoveTo(io);
}

}  // namespace pik
