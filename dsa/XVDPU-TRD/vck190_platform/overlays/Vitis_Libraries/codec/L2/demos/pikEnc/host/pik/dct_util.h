// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DCT_UTIL_H_
#define PIK_DCT_UTIL_H_

#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include <iostream>

namespace pik {

// Scatter/gather a SX*SY block into (SX/kBlockDim)*(SY/kBlockDim)
// kBlockDim*kBlockDim blocks. `block` should be composed of SY rows of SX
// contiguous blocks. In the output, each kBlockDim*kBlockDim block should be
// contiguous, and the same "block row" should be too, but different block rows
// are at a distance of `stride` pixels.
template <size_t SX, size_t SY, typename T>
void ScatterBlock(const T* PIK_RESTRICT block, size_t block_stride,
                  T* PIK_RESTRICT row, size_t stride) {
  constexpr size_t xblocks = SX / kBlockDim;
  constexpr size_t yblocks = SY / kBlockDim;
  for (size_t y = 0; y < SY; y++) {
    T* PIK_RESTRICT current_row =
        row + (y & (yblocks - 1)) * stride + (y / yblocks) * kBlockDim;
    for (size_t x = 0; x < SX; x++) {
      size_t block_pos = y * SX + x;
      size_t block_row = block_pos / (xblocks * kDCTBlockSize);
      size_t block_idx = block_pos & (xblocks * kDCTBlockSize - 1);
      current_row[(x & (xblocks - 1)) * kDCTBlockSize + (x / xblocks)] =
          block[block_row * block_stride + block_idx];

      std::cout<<"std_scatter out_y="<<(y & (yblocks - 1)) * stride + (y / yblocks) * kBlockDim<<" out_x="<<(x & (xblocks - 1)) * kDCTBlockSize + (x / xblocks)
    		  <<" in_y="<<block_row<<" in_x="<<block_idx<<" value="<<block[block_row * block_stride + block_idx]<<std::endl;
    }
  }
}

template <size_t SX, size_t SY, typename T>
void GatherBlock(const T* PIK_RESTRICT row, size_t stride,
                 T* PIK_RESTRICT block, size_t block_stride) {
  constexpr size_t xblocks = SX / kBlockDim;
  constexpr size_t yblocks = SY / kBlockDim;
  for (size_t y = 0; y < SY; y++) {
    const T* PIK_RESTRICT current_row =
        row + (y & (yblocks - 1)) * stride + (y / yblocks) * kBlockDim;
    for (size_t x = 0; x < SX; x++) {
      size_t block_pos = y * SX + x;
      size_t block_row = block_pos / (xblocks * kDCTBlockSize);
      size_t block_idx = block_pos & (xblocks * kDCTBlockSize - 1);
      block[block_row * block_stride + block_idx] =
          current_row[(x & (xblocks - 1)) * kDCTBlockSize + (x / xblocks)];
    }
  }
}

// Fills a preallocated (N*N)*W x H `dct` with (N*N)x1 blocks produced by
// ComputeTransposedScaledDCT() from the corresponding NxN block of
// `image`. Note that `dct` coefficients are scaled by 1 / (N*N), so that
// ComputeTransposedScaledIDCT applied to each block or TransposedScaledIDCT
// will return the original input.
// REQUIRES: image.xsize() == N*W, image.ysize() == N*H
SIMD_ATTR void TransposedScaledDCT(const Image3F& image,
                                   Image3F* PIK_RESTRICT dct);

// Fills a preallocated N*W x N*H `idct` with NxN blocks produced by
// ComputeTransposedScaledIDCT() from the (N*N)x1 blocks of `dct`.
// REQUIRES: dct.xsize() == N*N*W, dct.ysize() == H
SIMD_ATTR void TransposedScaledIDCT(const Image3F& dct,
                                    Image3F* PIK_RESTRICT idct);

// Returns an N x M image by taking the DC coefficient from each 64x1 block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
template <typename T>
Image3<T> DCImage(const Image3<T>& coeffs) {
  PIK_ASSERT(coeffs.xsize() % kDCTBlockSize == 0);
  Image3<T> out(coeffs.xsize() / kDCTBlockSize, coeffs.ysize());
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < out.ysize(); ++y) {
      const T* PIK_RESTRICT row_in = coeffs.ConstPlaneRow(c, y);
      T* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      for (size_t x = 0; x < out.xsize(); ++x) {
        row_out[x] = row_in[x * kDCTBlockSize];
      }
    }
  }
  return out;
}

// Scatters dc into "coeffs" at offset 0 within 1x64 blocks.
template <typename T>
void FillDC(const Image3<T>& dc, Image3<T>* PIK_RESTRICT coeffs) {
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < dc.ysize(); y++) {
      const T* PIK_RESTRICT row_dc = dc.ConstPlaneRow(c, y);
      T* PIK_RESTRICT row_out = coeffs->PlaneRow(c, y);
      for (size_t x = 0; x < dc.xsize(); ++x) {
        row_out[kDCTBlockSize * x] = row_dc[x];
      }
    }
  }
}

}  // namespace pik

#endif  // PIK_DCT_UTIL_H_
