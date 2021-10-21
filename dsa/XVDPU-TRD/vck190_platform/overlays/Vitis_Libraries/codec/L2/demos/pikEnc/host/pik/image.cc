// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/image.h"

#include <stdint.h>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/common.h"
#include "pik/profiler.h"


namespace pik {

CacheAlignedUniquePtr AllocateImageBytes(size_t size, size_t xsize,
                                         size_t ysize) {
  // (Can't profile CacheAligned itself because it is used by profiler.h)
  PROFILER_FUNC;


  // Note: size may be zero.
  CacheAlignedUniquePtr bytes = AllocateArray(size);
  PIK_ASSERT(reinterpret_cast<uintptr_t>(bytes.get()) % kImageAlign == 0);
  return bytes;
}

ImageB ImageFromPacked(const uint8_t* packed, const size_t xsize,
                       const size_t ysize, const size_t bytes_per_row) {
  PIK_ASSERT(bytes_per_row >= xsize);
  ImageB image(xsize, ysize);
  PROFILER_FUNC;
  for (size_t y = 0; y < ysize; ++y) {
    uint8_t* const PIK_RESTRICT row = image.Row(y);
    const uint8_t* const PIK_RESTRICT packed_row = packed + y * bytes_per_row;
    memcpy(row, packed_row, xsize);
  }
  return image;
}

// Note that using mirroring here gives slightly worse results.
Image3F PadImageToMultiple(const Image3F& in, const size_t N) {
  PROFILER_FUNC;
  const size_t xsize_blocks = DivCeil(in.xsize(), N);
  const size_t ysize_blocks = DivCeil(in.ysize(), N);
  const size_t xsize = N * xsize_blocks;
  const size_t ysize = N * ysize_blocks;
  Image3F out(xsize, ysize);
  for (int c = 0; c < 3; ++c) {
    int y = 0;
    for (; y < in.ysize(); ++y) {
      const float* PIK_RESTRICT row_in = in.ConstPlaneRow(c, y);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, in.xsize() * sizeof(row_in[0]));
      const int lastcol = in.xsize() - 1;
      const float lastval = row_out[lastcol];
      for (int x = in.xsize(); x < xsize; ++x) {
        row_out[x] = lastval;
      }
    }

    // TODO(janwas): no need to copy if we can 'extend' image: if rows are
    // pointers to any memory? Or allocate larger image before IO?
    const int lastrow = in.ysize() - 1;
    for (; y < ysize; ++y) {
      const float* PIK_RESTRICT row_in = out.ConstPlaneRow(c, lastrow);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
    }
  }
  return out;
}

float DotProduct(const ImageF& a, const ImageF& b) {
  double sum = 0.0;
  for (int y = 0; y < a.ysize(); ++y) {
    const float* const PIK_RESTRICT row_a = a.ConstRow(y);
    const float* const PIK_RESTRICT row_b = b.ConstRow(y);
    for (int x = 0; x < a.xsize(); ++x) {
      sum += row_a[x] * row_b[x];
    }
  }
  return sum;
}

}  // namespace pik
