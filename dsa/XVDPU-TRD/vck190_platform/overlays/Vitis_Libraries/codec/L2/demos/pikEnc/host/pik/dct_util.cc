// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/dct_util.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/bits.h"
#include "pik/common.h"
#include "pik/dct.h"
#include "pik/gauss_blur.h"
#include "pik/profiler.h"
#include "pik/simd/simd.h"
#include "pik/status.h"

namespace pik {

SIMD_ATTR void TransposedScaledDCT(const Image3F& image,
                                   Image3F* PIK_RESTRICT dct) {
  PROFILER_ZONE("TransposedScaledDCT facade");
  PIK_ASSERT(image.xsize() % kBlockDim == 0);
  PIK_ASSERT(image.ysize() % kBlockDim == 0);
  const size_t xsize_blocks = image.xsize() / kBlockDim;
  const size_t ysize_blocks = image.ysize() / kBlockDim;
  PIK_ASSERT(dct->xsize() == xsize_blocks * kDCTBlockSize);
  PIK_ASSERT(dct->ysize() == ysize_blocks);

  for (size_t c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_in = image.ConstPlaneRow(c, by * kBlockDim);
      float* PIK_RESTRICT row_dct = dct->PlaneRow(c, by);

      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        ComputeTransposedScaledDCT<kBlockDim>()(
            FromLines<kBlockDim>(row_in + bx * kBlockDim, image.PixelsPerRow()),
            ScaleToBlock<kBlockDim>(row_dct + bx * kDCTBlockSize));
      }
    }
  }
}

SIMD_ATTR void TransposedScaledIDCT(const Image3F& dct,
                                    Image3F* PIK_RESTRICT idct) {
  PROFILER_ZONE("IDCT facade");
  PIK_ASSERT(dct.xsize() % kDCTBlockSize == 0);
  const size_t xsize_blocks = dct.xsize() / kDCTBlockSize;
  const size_t ysize_blocks = dct.ysize();
  PIK_ASSERT(idct->xsize() == xsize_blocks * kBlockDim);
  PIK_ASSERT(idct->ysize() == ysize_blocks * kBlockDim);

  for (size_t c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_dct = dct.ConstPlaneRow(c, by);
      float* PIK_RESTRICT row_idct = idct->PlaneRow(c, by * kBlockDim);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        ComputeTransposedScaledIDCT<kBlockDim>()(
            FromBlock<kBlockDim>(row_dct + bx * kDCTBlockSize),
            ToLines<kBlockDim>(row_idct + bx * kBlockDim,
                               idct->PixelsPerRow()));
      }
    }
  }
}

}  // namespace pik
