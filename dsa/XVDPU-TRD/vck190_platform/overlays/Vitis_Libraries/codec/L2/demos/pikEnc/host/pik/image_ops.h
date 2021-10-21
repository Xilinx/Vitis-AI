// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_IMAGE_OPS_H_
#define PIK_IMAGE_OPS_H_

// Operations on images.

#include "pik/image.h"

namespace pik {


template <class ImageIn, class ImageOut>
void Subtract(const ImageIn& image1, const ImageIn& image2, ImageOut* out) {
  using T = typename ImageIn::T;
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());

  for (size_t y = 0; y < ysize; ++y) {
    const T* const PIK_RESTRICT row1 = image1.Row(y);
    const T* const PIK_RESTRICT row2 = image2.Row(y);
    T* const PIK_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = row1[x] - row2[x];
    }
  }
}

// In-place.
template <typename Tin, typename Tout>
void SubtractFrom(const Image<Tin>& what, Image<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (size_t y = 0; y < ysize; ++y) {
    const Tin* PIK_RESTRICT row_what = what.ConstRow(y);
    Tout* PIK_RESTRICT row_to = to->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_to[x] -= row_what[x];
    }
  }
}

// In-place.
template <typename Tin, typename Tout>
void AddTo(const Image<Tin>& what, Image<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (size_t y = 0; y < ysize; ++y) {
    const Tin* PIK_RESTRICT row_what = what.ConstRow(y);
    Tout* PIK_RESTRICT row_to = to->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_to[x] += row_what[x];
    }
  }
}

// Returns linear combination of two grayscale images.
template <typename T>
Image<T> LinComb(const T lambda1, const Image<T>& image1, const T lambda2,
                 const Image<T>& image2) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());
  Image<T> out(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const T* const PIK_RESTRICT row1 = image1.Row(y);
    const T* const PIK_RESTRICT row2 = image2.Row(y);
    T* const PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = lambda1 * row1[x] + lambda2 * row2[x];
    }
  }
  return out;
}

// Returns a pixel-by-pixel multiplication of image by lambda.
template <typename T>
Image<T> ScaleImage(const T lambda, const Image<T>& image) {
  Image<T> out(image.xsize(), image.ysize());
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    T* const PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      row_out[x] = lambda * row[x];
    }
  }
  return out;
}

template <typename T>
Image<T> Product(const Image<T>& a, const Image<T>& b) {
  Image<T> c(a.xsize(), a.ysize());
  for (size_t y = 0; y < a.ysize(); ++y) {
    const T* const PIK_RESTRICT row_a = a.Row(y);
    const T* const PIK_RESTRICT row_b = b.Row(y);
    T* const PIK_RESTRICT row_c = c.Row(y);
    for (size_t x = 0; x < a.xsize(); ++x) {
      row_c[x] = row_a[x] * row_b[x];
    }
  }
  return c;
}

float DotProduct(const ImageF& a, const ImageF& b);

template <typename T>
void FillImage(const T value, Image<T>* image) {
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      row[x] = value;
    }
  }
}

template <typename T>
void ZeroFillImage(Image<T>* image) {
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    memset(row, 0, image->xsize() * sizeof(T));
  }
}

// Generator for independent, uniformly distributed integers [0, max].
template <typename T, typename Random>
class GeneratorRandom {
 public:
  GeneratorRandom(Random* rng, const T max) : rng_(*rng), dist_(0, max) {}

  GeneratorRandom(Random* rng, const T min, const T max)
      : rng_(*rng), dist_(min, max) {}

  T operator()(const size_t x, const size_t y, const int c) const {
    return dist_(rng_);
  }

 private:
  Random& rng_;
  mutable std::uniform_int_distribution<> dist_;
};

template <typename Random>
class GeneratorRandom<float, Random> {
 public:
  GeneratorRandom(Random* rng, const float max)
      : rng_(*rng), dist_(0.0f, max) {}

  GeneratorRandom(Random* rng, const float min, const float max)
      : rng_(*rng), dist_(min, max) {}

  float operator()(const size_t x, const size_t y, const int c) const {
    return dist_(rng_);
  }

 private:
  Random& rng_;
  mutable std::uniform_real_distribution<float> dist_;
};

template <typename Random>
class GeneratorRandom<double, Random> {
 public:
  GeneratorRandom(Random* rng, const double max)
      : rng_(*rng), dist_(0.0, max) {}

  GeneratorRandom(Random* rng, const double min, const double max)
      : rng_(*rng), dist_(min, max) {}

  double operator()(const size_t x, const size_t y, const int c) const {
    return dist_(rng_);
  }

 private:
  Random& rng_;
  mutable std::uniform_real_distribution<> dist_;
};

// Assigns generator(x, y, 0) to each pixel (x, y).
template <class Generator, class Image>
void GenerateImage(const Generator& generator, Image* image) {
  using T = typename Image::T;
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      row[x] = generator(x, y, 0);
    }
  }
}

template <template <typename> class Image, typename T>
void RandomFillImage(Image<T>* image,
                     const T max = std::numeric_limits<T>::max()) {
  std::mt19937_64 rng(129);
  const GeneratorRandom<T, std::mt19937_64> generator(&rng, max);
  GenerateImage(generator, image);
}

template <template <typename> class Image, typename T>
void RandomFillImage(Image<T>* image, const T min, const T max,
                     const int seed) {
  std::mt19937_64 rng(seed);
  const GeneratorRandom<T, std::mt19937_64> generator(&rng, min, max);
  GenerateImage(generator, image);
}

// Mirrors out of bounds coordinates and returns valid coordinates unchanged.
// We assume the radius (distance outside the image) is small compared to the
// image size, otherwise this might not terminate.
// The mirror is outside the last column (border pixel is also replicated).
static inline int64_t Mirror(int64_t x, const int64_t xsize) {
  // TODO(janwas): replace with branchless version
  while (x < 0 || x >= xsize) {
    if (x < 0) {
      x = -x - 1;
    } else {
      x = 2 * xsize - 1 - x;
    }
  }
  return x;
}

// Wrap modes for ensuring X/Y coordinates are in the valid range [0, size):

// Mirrors (repeating the edge pixel once). Useful for convolutions.
struct WrapMirror {
  PIK_INLINE int64_t operator()(const int64_t coord, const int64_t size) const {
    return Mirror(coord, size);
  }
};

// Repeats the edge pixel.
struct WrapClamp {
  PIK_INLINE int64_t operator()(const int64_t coord, const int64_t size) const {
    return std::min(std::max<int64_t>(0, coord), size - 1);
  }
};

// Returns the same coordinate: required for TFNode with Border(), or useful
// when we know "coord" is already valid (e.g. interior of an image).
struct WrapUnchanged {
  PIK_INLINE int64_t operator()(const int64_t coord, const int64_t size) const {
    return coord;
  }
};

// Similar to Wrap* but for row pointers (reduces Row() multiplications).

class WrapRowMirror {
 public:
  template <class ImageOrView>
  WrapRowMirror(const ImageOrView& image, const size_t ysize)
      : first_row_(image.ConstRow(0)), last_row_(image.ConstRow(ysize - 1)) {}

  const float* const PIK_RESTRICT
  operator()(const float* const PIK_RESTRICT row, const int64_t stride) const {
    if (row < first_row_) {
      const int64_t num_before = first_row_ - row;
      // Mirrored; one row before => row 0, two before = row 1, ...
      return first_row_ + num_before - stride;
    }
    if (row > last_row_) {
      const int64_t num_after = row - last_row_;
      // Mirrored; one row after => last row, two after = last - 1, ...
      return last_row_ - num_after + stride;
    }
    return row;
  }

 private:
  const float* const PIK_RESTRICT first_row_;
  const float* const PIK_RESTRICT last_row_;
};

struct WrapRowUnchanged {
  PIK_INLINE const float* const PIK_RESTRICT
  operator()(const float* const PIK_RESTRICT row, const int64_t stride) const {
    return row;
  }
};

// Sets "thickness" pixels on each border to "value". This is faster than
// initializing the entire image and overwriting valid/interior pixels.
template <typename T>
void SetBorder(const size_t thickness, const T value, Image<T>* image) {
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  PIK_ASSERT(2 * thickness < xsize && 2 * thickness < ysize);
  // Top
  for (size_t y = 0; y < thickness; ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    std::fill(row, row + xsize, value);
  }

  // Bottom
  for (size_t y = ysize - thickness; y < ysize; ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    std::fill(row, row + xsize, value);
  }

  // Left/right
  for (size_t y = thickness; y < ysize - thickness; ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    std::fill(row, row + thickness, value);
    std::fill(row + xsize - thickness, row + xsize, value);
  }
}

// Computes the minimum and maximum pixel value.
template <typename T>
void ImageMinMax(const Image<T>& image, T* const PIK_RESTRICT min,
                 T* const PIK_RESTRICT max) {
  *min = std::numeric_limits<T>::max();
  *max = std::numeric_limits<T>::lowest();
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      *min = std::min(*min, row[x]);
      *max = std::max(*max, row[x]);
    }
  }
}

// Computes the average pixel value.
template <typename T>
double ImageAverage(const Image<T>& image) {
  double result = 0;
  size_t n = 0;
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      // Numerically stable method.
      double v = row[x];
      double delta = v - result;
      n++;
      result += delta / n;
    }
  }
  return result;
}

// Copies pixels, scaling their value relative to the "from" min/max by
// "to_range". Example: U8 [0, 255] := [0.0, 1.0], to_range = 1.0 =>
// outputs [0.0, 1.0].
template <typename FromType, typename ToType>
void ImageConvert(const Image<FromType>& from, const float to_range,
                  Image<ToType>* const PIK_RESTRICT to) {
  PIK_ASSERT(SameSize(from, *to));
  FromType min_from, max_from;
  ImageMinMax(from, &min_from, &max_from);
  const float scale = to_range / (max_from - min_from);
  for (size_t y = 0; y < from.ysize(); ++y) {
    const FromType* const PIK_RESTRICT row_from = from.Row(y);
    ToType* const PIK_RESTRICT row_to = to->Row(y);
    for (size_t x = 0; x < from.xsize(); ++x) {
      row_to[x] = static_cast<ToType>((row_from[x] - min_from) * scale);
    }
  }
}

// FromType and ToType are the pixel types.
template <typename FromType, typename ToType>
Image<ToType> StaticCastImage(const Image<FromType>& from) {
  Image<ToType> to(from.xsize(), from.ysize());
  for (size_t y = 0; y < from.ysize(); ++y) {
    const FromType* const PIK_RESTRICT row_from = from.Row(y);
    ToType* const PIK_RESTRICT row_to = to.Row(y);
    for (size_t x = 0; x < from.xsize(); ++x) {
      row_to[x] = static_cast<ToType>(row_from[x]);
    }
  }
  return to;
}


template <typename T>
Image<T> ImageFromPacked(const std::vector<T>& packed, const size_t xsize,
                         const size_t ysize) {
  Image<T> out(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    T* const PIK_RESTRICT row = out.Row(y);
    const T* const PIK_RESTRICT packed_row = &packed[y * xsize];
    memcpy(row, packed_row, xsize * sizeof(T));
  }
  return out;
}

// Computes independent minimum and maximum values for each plane.
template <typename T>
void Image3MinMax(const Image3<T>& image, const Rect& rect,
                  std::array<T, 3>* out_min, std::array<T, 3>* out_max) {
  for (int c = 0; c < 3; ++c) {
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();
    for (size_t y = 0; y < rect.ysize(); ++y) {
      const T* PIK_RESTRICT row = rect.ConstPlaneRow(image, c, y);
      for (size_t x = 0; x < rect.xsize(); ++x) {
        min = std::min(min, row[x]);
        max = std::max(max, row[x]);
      }
    }
    (*out_min)[c] = min;
    (*out_max)[c] = max;
  }
}

// Computes independent minimum and maximum values for each plane.
template <typename T>
void Image3MinMax(const Image3<T>& image, std::array<T, 3>* out_min,
                  std::array<T, 3>* out_max) {
  Image3MinMax(image, Rect(image), out_min, out_max);
}

template <typename T>
void Image3Max(const Image3<T>& image, std::array<T, 3>* out_max) {
  for (int c = 0; c < 3; ++c) {
    T max = std::numeric_limits<T>::min();
    for (size_t y = 0; y < image.ysize(); ++y) {
      const T* PIK_RESTRICT row = image.ConstPlaneRow(c, y);
      for (size_t x = 0; x < image.xsize(); ++x) {
        max = std::max(max, row[x]);
      }
    }
    (*out_max)[c] = max;
  }
}

// Computes the sum of the pixels in `rect`.
template <typename T>
T ImageSum(const Image<T>& image, const Rect& rect) {
  T result = 0;
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const T* PIK_RESTRICT row = rect.ConstRow(image, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      result += row[x];
    }
  }
  return result;
}

template <typename T>
T ImageSum(const Image<T>& image) {
  return ImageSum(image, Rect(image));
}

template <typename T>
std::array<T, 3> Image3Sum(const Image3<T>& image, const Rect& rect) {
  std::array<T, 3> out_sum = 0;
  for (int c = 0; c < 3; ++c) {
    (out_sum)[c] = ImageSum(image.Plane(c), rect);
  }
  return out_sum;
}

template <typename T>
std::array<T, 3> Image3Sum(const Image3<T>& image) {
  return Image3Sum(image, Rect(image));
}

template <typename T>
std::vector<T> PackedFromImage(const Image<T>& image, const Rect& rect) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  std::vector<T> packed(xsize * ysize);
  for (size_t y = 0; y < rect.ysize(); ++y) {
    memcpy(&packed[y * xsize], rect.ConstRow(image, y), xsize * sizeof(T));
  }
  return packed;
}

template <typename T>
std::vector<T> PackedFromImage(const Image<T>& image) {
  return PackedFromImage(image, Rect(image));
}

// Computes the median pixel value.
template <typename T>
T ImageMedian(const Image<T>& image, const Rect& rect) {
  std::vector<T> pixels = PackedFromImage(image, rect);
  return Median(&pixels);
}

template <typename T>
T ImageMedian(const Image<T>& image) {
  return ImageMedian(image, Rect(image));
}

template <typename T>
std::array<T, 3> Image3Median(const Image3<T>& image, const Rect& rect) {
  std::array<T, 3> out_median;
  for (int c = 0; c < 3; ++c) {
    (out_median)[c] = ImageMedian(image.Plane(c), rect);
  }
  return out_median;
}

template <typename T>
std::array<T, 3> Image3Median(const Image3<T>& image) {
  return Image3Median(image, Rect(image));
}

template <typename FromType, typename ToType>
void Image3Convert(const Image3<FromType>& from, const float to_range,
                   Image3<ToType>* const PIK_RESTRICT to) {
  PIK_ASSERT(SameSize(from, *to));
  std::array<FromType, 3> min_from, max_from;
  Image3MinMax(from, &min_from, &max_from);
  float scales[3];
  for (int c = 0; c < 3; ++c) {
    scales[c] = to_range / (max_from[c] - min_from[c]);
  }
  float scale = std::min(scales[0], std::min(scales[1], scales[2]));
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const FromType* PIK_RESTRICT row_from = from.ConstPlaneRow(c, y);
      ToType* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      for (size_t x = 0; x < from.xsize(); ++x) {
        const float to = (row_from[x] - min_from[c]) * scale;
        row_to[x] = static_cast<ToType>(to);
      }
    }
  }
}

// FromType and ToType are the pixel types.
template <typename ToType, typename FromType>
Image3<ToType> StaticCastImage3(const Image3<FromType>& from) {
  Image3<ToType> to(from.xsize(), from.ysize());
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const FromType* PIK_RESTRICT row_from = from.ConstPlaneRow(c, y);
      ToType* PIK_RESTRICT row_to = to.PlaneRow(c, y);
      for (size_t x = 0; x < from.xsize(); ++x) {
        row_to[x] = static_cast<ToType>(row_from[x]);
      }
    }
  }
  return to;
}

template <typename Tin, typename Tout>
void Subtract(const Image3<Tin>& image1, const Image3<Tin>& image2,
              Image3<Tout>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const Tin* const PIK_RESTRICT row1 = image1.ConstPlaneRow(c, y);
      const Tin* const PIK_RESTRICT row2 = image2.ConstPlaneRow(c, y);
      Tout* const PIK_RESTRICT row_out = out->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row1[x] - row2[x];
      }
    }
  }
}

template <typename Tin, typename Tout>
void SubtractFrom(const Image3<Tin>& what, Image3<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const Tin* PIK_RESTRICT row_what = what.ConstPlaneRow(c, y);
      Tout* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_to[x] -= row_what[x];
      }
    }
  }
}

template <typename Tin, typename Tout>
void AddTo(const Image3<Tin>& what, Image3<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const Tin* PIK_RESTRICT row_what = what.ConstPlaneRow(c, y);
      Tout* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_to[x] += row_what[x];
      }
    }
  }
}

template <typename T>
Image3<T> ScaleImage(const T lambda, const Image3<T>& image) {
  Image3<T> out(image.xsize(), image.ysize());
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image.ysize(); ++y) {
      const T* PIK_RESTRICT row = image.ConstPlaneRow(c, y);
      T* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      for (size_t x = 0; x < image.xsize(); ++x) {
        row_out[x] = lambda * row[x];
      }
    }
  }
  return out;
}

// Initializes all planes to the same "value".
template <typename T>
void FillImage(const T value, Image3<T>* image) {
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      for (size_t x = 0; x < image->xsize(); ++x) {
        row[x] = value;
      }
    }
  }
}

template <typename T>
void ZeroFillImage(Image3<T>* image) {
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      memset(row, 0, image->xsize() * sizeof(T));
    }
  }
}

template <typename T>
// TODO(firsching): add rect parameter to ZeroFillImage for consistency.
void ZeroFillImage(Image<T>* image, Rect rect) {
  for (size_t y = 0; y < rect.ysize(); ++y) {
    T* PIK_RESTRICT row = rect.Row(image, y);
    memset(row, 0, rect.xsize() * sizeof(T));
  }
}

// Assigns generator(x, y, c) to each pixel (x, y).
template <class Generator, typename T>
void GenerateImage(const Generator& generator, Image3<T>* image) {
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      for (size_t x = 0; x < image->xsize(); ++x) {
        row[x] = generator(x, y, c);
      }
    }
  }
}

template <template <typename> class Image, typename T>
void RandomFillImage(Image3<T>* image,
                     const T max = std::numeric_limits<T>::max()) {
  std::mt19937_64 rng(129);
  const GeneratorRandom<T, std::mt19937_64> generator(&rng, max);
  GenerateImage(generator, image);
}

template <template <typename> class Image, typename T>
void RandomFillImage(Image3<T>* image, const T min, const T max,
                     const int seed) {
  std::mt19937_64 rng(seed);
  const GeneratorRandom<T, std::mt19937_64> generator(&rng, min, max);
  GenerateImage(generator, image);
}

template <typename T>
std::vector<T> InterleavedFromImage3(const Image3<T>& image3) {
  const size_t xsize = image3.xsize();
  const size_t ysize = image3.ysize();
  std::vector<T> interleaved(xsize * ysize * 3);
  for (size_t y = 0; y < ysize; ++y) {
    const T* PIK_RESTRICT row0 = image3.ConstPlaneRow(0, y);
    const T* PIK_RESTRICT row1 = image3.ConstPlaneRow(1, y);
    const T* PIK_RESTRICT row2 = image3.ConstPlaneRow(2, y);
    T* const PIK_RESTRICT row_interleaved = &interleaved[y * xsize * 3];
    for (size_t x = 0; x < xsize; ++x) {
      row_interleaved[3 * x + 0] = row0[x];
      row_interleaved[3 * x + 1] = row1[x];
      row_interleaved[3 * x + 2] = row2[x];
    }
  }
  return interleaved;
}

template <typename T>
Image3<T> Image3FromInterleaved(const T* const interleaved, const size_t xsize,
                                const size_t ysize,
                                const size_t bytes_per_row) {
  PIK_ASSERT(bytes_per_row >= 3 * xsize * sizeof(T));
  Image3<T> image3(xsize, ysize);
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(interleaved);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      T* PIK_RESTRICT row_out = image3.PlaneRow(c, y);
      const T* row_interleaved =
          reinterpret_cast<const T*>(bytes + y * bytes_per_row);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row_interleaved[3 * x + c];
      }
    }
  }
  return image3;
}

// First, image is padded horizontally, with the rightmost value.
// Next, image is padded vertically, by repeating the last line.
Image3F PadImageToMultiple(const Image3F& in, const size_t N);

}  // namespace pik

#endif  // PIK_IMAGE_OPS_H_
