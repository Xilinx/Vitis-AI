// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_IMAGE_H_
#define PIK_IMAGE_H_

// SIMD/multicore-friendly planar image representation with row accessors.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "pik/cache_aligned.h"
#include "pik/compiler_specific.h"
#include "pik/profiler.h"
#include "pik/robust_statistics.h"
#include "pik/status.h"

namespace pik {

// Each row address is a multiple of this - enables aligned loads.
static constexpr size_t kImageAlign = CacheAligned::kAlignment;
static_assert(kImageAlign >= kMaxVectorSize, "Insufficient alignment");

// Returns distance [bytes] between the start of two consecutive rows, a
// multiple of kAlign but NOT CacheAligned::kAlias - see below.
//
// Differing "kAlign" make sense for:
// - Image: 128 to avoid false sharing/RFOs between multiple threads processing
//   rows independently;
// - TileFlow: no cache line alignment needed because buffers are per-thread;
//   just need kMaxVectorSize=16..64 for SIMD.
//
// "valid_bytes" is xsize * sizeof(T).
template <size_t kAlign>
static inline size_t BytesPerRow(const size_t valid_bytes) {
  static_assert((kAlign & (kAlign - 1)) == 0, "kAlign should be power of two");

  // Extra two vectors allow *writing* a partial or full vector on the right AND
  // left border (for convolve.h) without disturbing the next/previous row.
  const size_t row_size = valid_bytes + 2 * kMaxVectorSize;

  // Round up.
  size_t bytes_per_row = (row_size + kAlign - 1) & ~(kAlign - 1);

  // During the lengthy window before writes are committed to memory, CPUs
  // guard against read after write hazards by checking the address, but
  // only the lower 11 bits. We avoid a false dependency between writes to
  // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
  // Avoid2K prevents the same problem for the planes of an Image3.
  if (bytes_per_row % CacheAligned::kAlias == 0) {
    bytes_per_row += kImageAlign;
  }

  return bytes_per_row;
}

// Factored out of Image<> to avoid dependency on profiler.h and <atomic>.
CacheAlignedUniquePtr AllocateImageBytes(size_t size, size_t xsize,
                                         size_t ysize);

// Single channel, aligned rows separated by padding. T must be POD.
//
// Rationale: vectorization benefits from aligned operands - unaligned loads and
// especially stores are expensive when the address crosses cache line
// boundaries. Introducing padding after each row ensures the start of a row is
// aligned, and that row loops can process entire vectors (writes to the padding
// are allowed and ignored).
//
// We prefer a planar representation, where channels are stored as separate
// 2D arrays, because that simplifies vectorization (repeating the same
// operation on multiple adjacent components) without the complexity of a
// hybrid layout (8 R, 8 G, 8 B, ...). In particular, clients can easily iterate
// over all components in a row and Image requires no knowledge of the pixel
// format beyond the component type "T".
//
// This image layout could also be achieved with a vector and a row accessor
// function, but a class wrapper with support for "deleter" allows wrapping
// existing memory allocated by clients without copying the pixels. It also
// provides convenient accessors for xsize/ysize, which shortens function
// argument lists. Supports move-construction so it can be stored in containers.
template <typename ComponentType>
class Image {
 public:
  using T = ComponentType;
  static constexpr size_t kNumPlanes = 1;

  Image() : xsize_(0), ysize_(0), bytes_per_row_(0), bytes_(nullptr) {}

  Image(const size_t xsize, const size_t ysize)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(BytesPerRow<kImageAlign>(xsize * sizeof(T))),
        bytes_(nullptr) {
    PIK_ASSERT(bytes_per_row_ % kImageAlign == 0);
    // xsize and/or ysize can legitimately be zero, in which case we don't
    // want to allocate.
    if (xsize != 0 && ysize != 0) {
      bytes_ = AllocateImageBytes(bytes_per_row_ * ysize + kMaxVectorSize,
                                  xsize, ysize);
    }

#ifdef MEMORY_SANITIZER
    // Only in MSAN builds: ensure full vectors are initialized.
    const size_t partial = (xsize_ * sizeof(T)) % kMaxVectorSize;
    const size_t remainder = (partial == 0) ? 0 : (kMaxVectorSize - partial);
    for (size_t y = 0; y < ysize_; ++y) {
      memset(Row(y) + xsize_, 0, remainder);
    }
#endif
  }

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo() instead.
  Image(const Image& other) = delete;
  Image& operator=(const Image& other) = delete;

  // Move constructor (required for returning Image from function)
  Image(Image&& other) = default;

  // Move assignment (required for std::vector)
  Image& operator=(Image&& other) = default;

  void Swap(Image& other) {
    std::swap(xsize_, other.xsize_);
    std::swap(ysize_, other.ysize_);
    std::swap(bytes_per_row_, other.bytes_per_row_);
    std::swap(bytes_, other.bytes_);
  }

  // Useful for pre-allocating image with some padding for alignment purposes
  // and later reporting the actual valid dimensions. Caller is responsible
  // for ensuring xsize/ysize are <= the original dimensions.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
    xsize_ = static_cast<uint32_t>(xsize);
    ysize_ = static_cast<uint32_t>(ysize);
    // NOTE: we can't recompute bytes_per_row for more compact storage and
    // better locality because that would invalidate the image contents.
  }

  // How many pixels.
  PIK_INLINE size_t xsize() const { return xsize_; }
  PIK_INLINE size_t ysize() const { return ysize_; }

  // Returns pointer to the start of a row, with at least xsize (rounded up to
  // kImageAlign bytes) accessible values.
  PIK_INLINE T* PIK_RESTRICT Row(const size_t y) {
    RowBoundsCheck(y);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to non-const - required for writing to individual planes
  // of an Image3.
  PIK_INLINE T* PIK_RESTRICT MutableRow(const size_t y) const {
    RowBoundsCheck(y);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above).
  PIK_INLINE const T* PIK_RESTRICT Row(const size_t y) const {
    RowBoundsCheck(y);
    const void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<const T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above), even if called on a non-const Image.
  PIK_INLINE const T* PIK_RESTRICT ConstRow(const size_t y) const {
    return Row(y);
  }

  // Raw access to byte contents, for interfacing with other libraries.
  // Unsigned char instead of char to avoid surprises (sign extension).
  PIK_INLINE uint8_t* PIK_RESTRICT bytes() {
    void* p = bytes_.get();
    return static_cast<uint8_t * PIK_RESTRICT>(PIK_ASSUME_ALIGNED(p, 64));
  }
  PIK_INLINE const uint8_t* PIK_RESTRICT bytes() const {
    const void* p = bytes_.get();
    return static_cast<const uint8_t * PIK_RESTRICT>(PIK_ASSUME_ALIGNED(p, 64));
  }

  // NOTE: do not use this for copying rows - the valid xsize may be much less.
  PIK_INLINE size_t bytes_per_row() const { return bytes_per_row_; }

  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must
  // NOT be used to determine xsize. NOTE: this is less efficient than
  // ByteOffset(row, bytes_per_row).
  PIK_INLINE intptr_t PixelsPerRow() const {
    static_assert(kImageAlign % sizeof(T) == 0,
                  "Padding must be divisible by the pixel size.");
    return static_cast<intptr_t>(bytes_per_row_ / sizeof(T));
  }

 private:
  PIK_INLINE void RowBoundsCheck(const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER)
    if (y >= ysize_) {
      Abort(__FILE__, __LINE__, "Row(%zu) >= %zu\n", y, ysize_);
    }
#endif
  }

  // (Members are non-const to enable assignment during move-assignment.)
  uint32_t xsize_;  // In valid pixels, not including any padding.
  uint32_t ysize_;
  size_t bytes_per_row_;  // Includes padding.
  CacheAlignedUniquePtr bytes_;
};

using ImageB = Image<uint8_t>;
using ImageS = Image<int16_t>;  // signed integer or half-float
using ImageU = Image<uint16_t>;
using ImageI = Image<int32_t>;
using ImageF = Image<float>;
using ImageD = Image<double>;

// We omit unnecessary fields and choose smaller representations to reduce L1
// cache pollution.
#pragma pack(push, 1)

// Size of an image in pixels. POD.
struct ImageSize {
  static ImageSize Make(const size_t xsize, const size_t ysize) {
    ImageSize ret;
    ret.xsize = static_cast<uint32_t>(xsize);
    ret.ysize = static_cast<uint32_t>(ysize);
    return ret;
  }

  bool operator==(const ImageSize& other) const {
    return xsize == other.xsize && ysize == other.ysize;
  }

  uint32_t xsize;
  uint32_t ysize;
};

#pragma pack(pop)

template <typename T>
void CopyImageTo(const Image<T>& from, Image<T>* PIK_RESTRICT to) {
  PROFILER_ZONE("CopyImage1");
  PIK_ASSERT(SameSize(from, *to));
  for (size_t y = 0; y < from.ysize(); ++y) {
    const T* PIK_RESTRICT row_from = from.ConstRow(y);
    T* PIK_RESTRICT row_to = to->Row(y);
    memcpy(row_to, row_from, from.xsize() * sizeof(T));
  }
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image<T> CopyImage(const Image<T>& from) {
  Image<T> to(from.xsize(), from.ysize());
  CopyImageTo(from, &to);
  return to;
}

// Also works for Image3 and mixed argument types.
template <class Image1, class Image2>
bool SameSize(const Image1& image1, const Image2& image2) {
  return image1.xsize() == image2.xsize() && image1.ysize() == image2.ysize();
}

template <typename T>
bool SamePixels(const Image<T>& image1, const Image<T>& image2) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const PIK_RESTRICT row1 = image1.Row(y);
    const T* const PIK_RESTRICT row2 = image2.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      if (row1[x] != row2[x]) {
        return false;
      }
    }
  }
  return true;
}

// Use for floating-point images with fairly large numbers; tolerates small
// absolute errors and/or small relative errors. Returns max_relative.
template <typename T>
double VerifyRelativeError(const Image<T>& expected, const Image<T>& actual,
                           const double threshold_l1,
                           const double threshold_relative,
                           const size_t border = 0, const size_t c = 0) {
  PIK_CHECK(SameSize(expected, actual));
  // Max over current scanline to give a better idea whether there are
  // systematic errors or just one outlier. Invalid if negative.
  double max_l1 = -1;
  double max_relative = -1;
  for (size_t y = border; y < expected.ysize() - border; ++y) {
    const T* const PIK_RESTRICT row_expected = expected.Row(y);
    const T* const PIK_RESTRICT row_actual = actual.Row(y);
    bool any_bad = false;
    for (size_t x = border; x < expected.xsize() - border; ++x) {
      const double l1 = std::abs(row_expected[x] - row_actual[x]);

      // Cannot compute relative, only check/update L1.
      if (row_expected[x] < 1E-10) {
        if (l1 > threshold_l1) {
          any_bad = true;
          max_l1 = std::max(max_l1, l1);
        }
      } else {
        const double relative = l1 / std::abs(double(row_expected[x]));
        if (l1 > threshold_l1 && relative > threshold_relative) {
          // Fails both tolerances => will exit below, update max_*.
          any_bad = true;
          max_l1 = std::max(max_l1, l1);
          max_relative = std::max(max_relative, relative);
        }
      }
    }

    if (any_bad) {
      // Never had a valid relative value, don't print it.
      if (max_relative < 0) {
        printf("c=%zu: max +/- %E exceeds +/- %.2E\n", c, max_l1, threshold_l1);
      } else {
        printf("c=%zu: max +/- %E, x %E exceeds +/- %.2E, x %.2E\n", c, max_l1,
               max_relative, threshold_l1, threshold_relative);
      }
      // Find first failing x for further debugging.
      for (size_t x = border; x < expected.xsize() - border; ++x) {
        const double l1 = std::abs(row_expected[x] - row_actual[x]);

        bool bad = l1 > threshold_l1;
        if (row_expected[x] > 1E-10) {
          const double relative = l1 / std::abs(double(row_expected[x]));
          bad &= relative > threshold_relative;
        }
        if (bad) {
          PIK_ABORT("%zu, %zu (%zu x %zu) expected %f actual %f\n", x, y,
                    expected.xsize(), expected.ysize(),
                    static_cast<double>(row_expected[x]),
                    static_cast<double>(row_actual[x]));
        }
      }

      PIK_CHECK(false);  // if any_bad, we should have exited.
    }
  }

  return (max_relative < 0) ? 0.0 : max_relative;
}

template <typename T>
class Image3;

// Rectangular region in image(s). Factoring this out of Image instead of
// shifting the pointer by x0/y0 allows this to apply to multiple images with
// different resolutions (e.g. color transform and quantization field).
// Can compare using SameSize(rect1, rect2).
class Rect {
 public:
  // Most windows are xsize_max * ysize_max, except those on the borders where
  // begin + size_max > end.
  constexpr Rect(size_t xbegin, size_t ybegin, size_t xsize_max,
                 size_t ysize_max, size_t xend, size_t yend)
      : x0_(xbegin),
        y0_(ybegin),
        xsize_(ClampedSize(xbegin, xsize_max, xend)),
        ysize_(ClampedSize(ybegin, ysize_max, yend)) {}

  // Construct with origin and known size (typically from another Rect).
  constexpr Rect(size_t xbegin, size_t ybegin, size_t xsize, size_t ysize)
      : x0_(xbegin), y0_(ybegin), xsize_(xsize), ysize_(ysize) {}

  // Construct a rect that covers a whole image
  template <typename T>
  explicit Rect(const Image3<T>& image)
      : Rect(0, 0, image.xsize(), image.ysize()) {}
  template <typename T>
  explicit Rect(const Image<T>& image)
      : Rect(0, 0, image.xsize(), image.ysize()) {}

  Rect(const Rect&) = default;
  Rect& operator=(const Rect&) = default;

  Rect Subrect(size_t xbegin, size_t ybegin, size_t xsize_max,
               size_t ysize_max) {
    return Rect(x0_ + xbegin, y0_ + ybegin, xsize_max, ysize_max, x0_ + xsize_,
                y0_ + ysize_);
  }

  template <typename T>
  T* Row(Image<T>* image, size_t y) const {
    return image->Row(y + y0_) + x0_;
  }

  template <typename T>
  T* PlaneRow(Image3<T>* image, const int c, size_t y) const {
    return image->PlaneRow(c, y + y0_) + x0_;
  }

  template <typename T>
  const T* ConstRow(const Image<T>& image, size_t y) const {
    return image.ConstRow(y + y0_) + x0_;
  }

  template <typename T>
  const T* ConstPlaneRow(const Image3<T>& image, const int c, size_t y) const {
    return image.ConstPlaneRow(c, y + y0_) + x0_;
  }

  // Returns true if this Rect fully resides in the given image. ImageT could be
  // Image<T> or Image3<T>; however if ImageT is Rect, results are nonsensical.
  template <class ImageT>
  bool IsInside(const ImageT& image) const {
    return (x0_ + xsize_ <= image.xsize()) && (y0_ + ysize_ <= image.ysize());
  }

  size_t x0() const { return x0_; }
  size_t y0() const { return y0_; }
  size_t xsize() const { return xsize_; }
  size_t ysize() const { return ysize_; }

 private:
  // Returns size_max, or whatever is left in [begin, end).
  static constexpr size_t ClampedSize(size_t begin, size_t size_max,
                                      size_t end) {
    return (begin + size_max <= end) ? size_max : end - begin;
  }

  size_t x0_;
  size_t y0_;

  size_t xsize_;
  size_t ysize_;
};

// Copies `from:rect` to `to`.
template <typename T>
void CopyImageTo(const Rect& rect, const Image<T>& from,
                 Image<T>* PIK_RESTRICT to) {
  PROFILER_ZONE("CopyImageR");
  PIK_ASSERT(SameSize(rect, *to));
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const T* PIK_RESTRICT row_from = rect.ConstRow(from, y);
    T* PIK_RESTRICT row_to = to->Row(y);
    memcpy(row_to, row_from, rect.xsize() * sizeof(T));
  }
}

// DEPRECATED - Returns a copy of the "image" pixels that lie in "rect".
template <typename T>
Image<T> CopyImage(const Rect& rect, const Image<T>& image) {
  Image<T> copy(rect.xsize(), rect.ysize());
  CopyImageTo(rect, image, &copy);
  return copy;
}

// Currently, we abuse Image to either refer to an image that owns its storage
// or one that doesn't. In similar vein, we abuse Image* function parameters to
// either mean "assign to me" or "fill the provided image with data".
// Hopefully, the "assign to me" meaning will go away and most images in the Pik
// codebase will not be backed by own storage. When this happens we can redesign
// Image to be a non-storage-holding view class and introduce BackedImage in
// those places that actually need it.

// NOTE: we can't use Image as a view because invariants are violated
// (alignment and the presence of padding before/after each "row").

// A bundle of 3 same-sized images. Typically constructed by moving from three
// rvalue references to Image. To overwrite an existing Image3 using
// single-channel producers, we also need access to Image*. Constructing
// temporary non-owning Image pointing to one plane of an existing Image3 risks
// dangling references, especially if the wrapper is moved. Therefore, we
// store an array of Image (which are compact enough that size is not a concern)
// and provide a Plane+MutableRow accessors.
template <typename ComponentType>
class Image3 {
 public:
  using T = ComponentType;
  using PlaneT = Image<T>;
  static constexpr size_t kNumPlanes = 3;

  Image3() : planes_{PlaneT(), PlaneT(), PlaneT()} {}

  Image3(const size_t xsize, const size_t ysize)
      : planes_{PlaneT(xsize, ysize), PlaneT(xsize, ysize),
                PlaneT(xsize, ysize)} {}

  Image3(Image3&& other) {
    for (int i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
  }

  Image3(PlaneT&& plane0, PlaneT&& plane1, PlaneT&& plane2) {
    PIK_CHECK(SameSize(plane0, plane1));
    PIK_CHECK(SameSize(plane0, plane2));
    planes_[0] = std::move(plane0);
    planes_[1] = std::move(plane1);
    planes_[2] = std::move(plane2);
  }

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo instead.
  Image3(const Image3& other) = delete;
  Image3& operator=(const Image3& other) = delete;

  Image3& operator=(Image3&& other) {
    for (int i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
    return *this;
  }

  // Returns row pointer; usage: PlaneRow(idx_plane, y)[x] = val.
  PIK_INLINE T* PIK_RESTRICT PlaneRow(const size_t c, const size_t y) {
    // Custom implementation instead of calling planes_[c].Row ensures only a
    // single multiplication is needed for PlaneRow(0..2, y).
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    void* row = planes_[c].bytes() + row_offset;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer; usage: val = PlaneRow(idx_plane, y)[x].
  PIK_INLINE const T* PIK_RESTRICT PlaneRow(const size_t c,
                                            const size_t y) const {
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    const void* row = planes_[c].bytes() + row_offset;
    return static_cast<const T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer, even if called from a non-const Image3.
  PIK_INLINE const T* PIK_RESTRICT ConstPlaneRow(const size_t c,
                                                 const size_t y) const {
    return PlaneRow(c, y);
  }

  PIK_INLINE const PlaneT& Plane(size_t idx) const { return planes_[idx]; }

  void Swap(Image3& other) {
    for (int c = 0; c < 3; ++c) {
      other.planes_[c].Swap(planes_[c]);
    }
  }

  void ShrinkTo(const size_t xsize, const size_t ysize) {
    for (PlaneT& plane : planes_) {
      plane.ShrinkTo(xsize, ysize);
    }
  }

  // Sizes of all three images are guaranteed to be equal.
  PIK_INLINE size_t xsize() const { return planes_[0].xsize(); }
  PIK_INLINE size_t ysize() const { return planes_[0].ysize(); }
  // Returns offset [bytes] from one row to the next row of the same plane.
  // WARNING: this must NOT be used to determine xsize, nor for copying rows -
  // the valid xsize may be much less.
  PIK_INLINE size_t bytes_per_row() const { return planes_[0].bytes_per_row(); }
  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must NOT be used
  // to determine xsize. NOTE: this is less efficient than
  // ByteOffset(row, bytes_per_row).
  PIK_INLINE intptr_t PixelsPerRow() const { return planes_[0].PixelsPerRow(); }

 private:
  PIK_INLINE void PlaneRowBoundsCheck(const size_t c, const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER)
    if (c >= kNumPlanes || y >= ysize()) {
      Abort(__FILE__, __LINE__, "PlaneRow(%zu, %zu) >= %zu\n", c, y, ysize());
    }
#endif
  }

 private:
  PlaneT planes_[kNumPlanes];
};

using Image3B = Image3<uint8_t>;
using Image3S = Image3<int16_t>;
using Image3U = Image3<uint16_t>;
using Image3I = Image3<int32_t>;
using Image3F = Image3<float>;
using Image3D = Image3<double>;

template <typename T>
void CopyImageTo(const Image3<T>& from, Image3<T>* PIK_RESTRICT to) {
  PROFILER_ZONE("CopyImage3");
  PIK_ASSERT(SameSize(from, *to));

  for (size_t c = 0; c < from.kNumPlanes; ++c) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const T* PIK_RESTRICT row_from = from.ConstPlaneRow(c, y);
      T* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      memcpy(row_to, row_from, from.xsize() * sizeof(T));
    }
  }
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image3<T> CopyImage(const Image3<T>& from) {
  Image3<T> copy(from.xsize(), from.ysize());
  CopyImageTo(from, &copy);
  return copy;
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image3<T> CopyImage(const Rect& rect, const Image3<T>& from) {
  Image3<T> to(rect.xsize(), rect.ysize());
  CopyImageTo(rect, from.Plane(0), const_cast<ImageF*>(&to.Plane(0)));
  CopyImageTo(rect, from.Plane(1), const_cast<ImageF*>(&to.Plane(1)));
  CopyImageTo(rect, from.Plane(2), const_cast<ImageF*>(&to.Plane(2)));
  return to;
}

template <typename T>
bool SamePixels(const Image3<T>& image1, const Image3<T>& image2) {
  PIK_CHECK(SameSize(image1, image2));
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const T* PIK_RESTRICT row1 = image1.PlaneRow(c, y);
      const T* PIK_RESTRICT row2 = image2.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        if (row1[x] != row2[x]) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename T>
double VerifyRelativeError(const Image3<T>& expected, const Image3<T>& actual,
                           const float threshold_l1,
                           const float threshold_relative,
                           const size_t border = 0) {
  double max_relative = 0.0;
  for (int c = 0; c < 3; ++c) {
    const double rel =
        VerifyRelativeError(expected.Plane(c), actual.Plane(c), threshold_l1,
                            threshold_relative, border, c);
    max_relative = std::max(max_relative, rel);
  }
  return max_relative;
}

// Sets "thickness" pixels on each border to "value". This is faster than
// initializing the entire image and overwriting valid/interior pixels.
template <typename T>
void SetBorder(const size_t thickness, const T value, Image3<T>* image) {
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  PIK_ASSERT(2 * thickness < xsize && 2 * thickness < ysize);
  // Top
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < thickness; ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      std::fill(row, row + xsize, value);
    }

    // Bottom
    for (size_t y = ysize - thickness; y < ysize; ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      std::fill(row, row + xsize, value);
    }

    // Left/right
    for (size_t y = thickness; y < ysize - thickness; ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      std::fill(row, row + thickness, value);
      std::fill(row + xsize - thickness, row + xsize, value);
    }
  }
}


}  // namespace pik

#endif  // PIK_IMAGE_H_
