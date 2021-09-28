// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/dct_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pik/codec.h"
#include "pik/dct.h"
#include "pik/entropy_coder.h"
#include "pik/gauss_blur.h"
#include "pik/image_test_utils.h"
#include "pik/opsin_image.h"
#include "pik/status.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

// Zeroes out the top-left 2x2 corner of each DCT block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
static void ZeroOut2x2(Image3F* coeffs) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  PIK_ASSERT(coeffs->xsize() % block_size == 0);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < coeffs->ysize(); ++y) {
      float* PIK_RESTRICT row = coeffs->PlaneRow(c, y);
      for (size_t x = 0; x < coeffs->xsize(); x += block_size) {
        row[x] = row[x + 1] = row[x + N] = row[x + N + 1] = 0.0f;
      }
    }
  }
}

Image3F KeepOnly2x2Corners(const Image3F& coeffs) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  Image3F copy = CopyImage(coeffs);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < coeffs.ysize(); ++y) {
      float* PIK_RESTRICT row = copy.PlaneRow(c, y);
      // TODO(user): might be better to copy 4 values, zero-out, place-back.
      for (size_t x = 0; x < coeffs.xsize(); x += block_size) {
        for (int k = 0; k < block_size; ++k) {
          if ((k >= (2 * N)) || ((k % N) >= 2)) row[x + k] = 0.0f;
        }
      }
    }
  }
  return copy;
}

// Returns a 2*N x 2*M image which is defined by the following 3 transforms:
//  1) zero out every coefficient that is outside the top 2x2 corner
//  2) apply ComputeTransposedScaledIDCT() to every block
//  3) subsample the result 4x4 by taking simple averages
// REQUIRES: coeffs.xsize() == kBlockSize*N, coeffs.ysize() == M
static Image3F GetPixelSpaceImageFrom0HVD_64(const Image3F& coeffs) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  PIK_ASSERT(coeffs.xsize() % block_size == 0);
  const size_t block_xsize = coeffs.xsize() / block_size;
  const size_t block_ysize = coeffs.ysize();
  Image3F out(block_xsize * 2, block_ysize * 2);
  // TODO(user): what is this magic constant?
  const float magic = 0.9061274463528878f;  // sin(7 * pi / 16) * cos(pi / 8)
  const float kScale01 = N * magic * DCTScales<N>()[0] * DCTScales<N>()[1];
  const float kScale11 =
      N * magic * magic * DCTScales<N>()[1] * DCTScales<N>()[1];
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < block_ysize; ++by) {
      const float* PIK_RESTRICT row_coeffs = coeffs.PlaneRow(c, by);
      float* PIK_RESTRICT row_out0 = out.PlaneRow(c, 2 * by + 0);
      float* PIK_RESTRICT row_out1 = out.PlaneRow(c, 2 * by + 1);
      for (size_t bx = 0; bx < block_xsize; ++bx) {
        const float* block = row_coeffs + bx * block_size;
        const float a00 = block[0];
        const float a01 = block[N] * kScale01;
        const float a10 = block[1] * kScale01;
        const float a11 = block[N + 1] * kScale11;
        row_out0[2 * bx + 0] = a00 + a01 + a10 + a11;
        row_out0[2 * bx + 1] = a00 - a01 + a10 - a11;
        row_out1[2 * bx + 0] = a00 + a01 - a10 - a11;
        row_out1[2 * bx + 1] = a00 - a01 - a10 + a11;
      }
    }
  }
  return out;
}

// Puts back the top 2x2 corner of each 8x8 block of *coeffs from the
// transformed pixel space image img.
// REQUIRES: coeffs->xsize() == 64*N, coeffs->ysize() == M
static void Add2x2CornersFromPixelSpaceImage(const Image3F& img,
                                             Image3F* coeffs) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  PIK_ASSERT(coeffs->xsize() % block_size == 0);
  const size_t block_xsize = coeffs->xsize() / block_size;
  const size_t block_ysize = coeffs->ysize();
  PIK_ASSERT(block_xsize * 2 <= img.xsize());
  PIK_ASSERT(block_ysize * 2 <= img.ysize());
  // TODO(user): what is this magic constant?
  const float magic = 0.9061274463528878f;  // sin(7 * pi / 16) * cos(pi / 8)
  const float kScale01 = N * magic * DCTScales<N>()[0] * DCTScales<N>()[1];
  const float kScale11 =
      N * magic * magic * DCTScales<N>()[1] * DCTScales<N>()[1];
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < block_ysize; ++by) {
      const float* PIK_RESTRICT row0 = img.PlaneRow(c, 2 * by + 0);
      const float* PIK_RESTRICT row1 = img.PlaneRow(c, 2 * by + 1);
      float* row_out = coeffs->PlaneRow(c, by);
      for (size_t bx = 0; bx < block_xsize; ++bx) {
        const float b00 = row0[2 * bx + 0];
        const float b01 = row0[2 * bx + 1];
        const float b10 = row1[2 * bx + 0];
        const float b11 = row1[2 * bx + 1];
        const float a00 = 0.25f * (b00 + b01 + b10 + b11);
        const float a01 = 0.25f * (b00 - b01 + b10 - b11);
        const float a10 = 0.25f * (b00 + b01 - b10 - b11);
        const float a11 = 0.25f * (b00 - b01 - b10 + b11);
        float* PIK_RESTRICT block = &row_out[bx * block_size];
        block[0] = a00;
        block[1] = a10 / kScale01;
        block[N] = a01 / kScale01;
        block[N + 1] = a11 / kScale11;
      }
    }
  }
}

// Returns an N x M image where each pixel is the average of the corresponding
// f x f block in the original.
// REQUIRES: image.xsize() == f*N, image.ysize() == f *M
static Image3F Subsample(const Image3F& image, int f) {
  PIK_CHECK(image.xsize() % f == 0);
  PIK_CHECK(image.ysize() % f == 0);
  const int shift = CeilLog2Nonzero(static_cast<uint32_t>(f));
  PIK_CHECK(f == (1 << shift));
  const size_t nxs = image.xsize() >> shift;
  const size_t nys = image.ysize() >> shift;
  Image3F retval(nxs, nys);
  ZeroFillImage(&retval);
  const float mul = 1.0f / (f * f);
  for (size_t y = 0; y < image.ysize(); ++y) {
    const float* row_in[3] = {image.PlaneRow(0, y), image.PlaneRow(1, y),
                              image.PlaneRow(2, y)};
    const size_t ny = y >> shift;
    float* row_out[3] = {retval.PlaneRow(0, ny), retval.PlaneRow(1, ny),
                         retval.PlaneRow(2, ny)};
    for (size_t c = 0; c < 3; ++c) {
      for (size_t x = 0; x < image.xsize(); ++x) {
        size_t nx = x >> shift;
        row_out[c][nx] += mul * row_in[c][x];
      }
    }
  }
  return retval;
}

static Image3F OpsinTestImage() {
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecContext codec_context;
  CodecInOut io(&codec_context);
  PIK_CHECK(io.SetFromFile(pathname, /*pool=*/nullptr));
  Image3F opsin = OpsinDynamicsImage(&io, Rect(io.color()));
  opsin.ShrinkTo(opsin.ysize() & ~7, opsin.xsize() & ~7);
  return opsin;
}

static SIMD_ATTR Image3F TransposedScaledIDCT(const Image3F& coeffs) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  const size_t xsize_blocks = coeffs.xsize() / block_size;
  const size_t ysize_blocks = coeffs.ysize();
  Image3F img(xsize_blocks * N, ysize_blocks * N);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const size_t stride = img.PlaneRow(c, 1) - img.PlaneRow(c, 0);
        const float* PIK_RESTRICT row_in = coeffs.PlaneRow(c, by);
        float* PIK_RESTRICT row_out = img.PlaneRow(c, by * N);
        ComputeTransposedScaledIDCT<N>()(FromBlock<N>(row_in + bx * block_size),
                                         ToLines<N>(row_out + bx * N, stride));
      }
    }
  }
  return img;
}

TEST(DctUtilTest, DCTRoundtrip) {
  constexpr size_t N = kBlockDim;
  constexpr int block_size = N * N;
  Image3F opsin = OpsinTestImage();
  const size_t xsize_blocks = opsin.xsize() / N;
  const size_t ysize_blocks = opsin.ysize() / N;

  Image3F coeffs(xsize_blocks * block_size, ysize_blocks);
  TransposedScaledDCT(opsin, &coeffs);
  Image3F recon = TransposedScaledIDCT(coeffs);

  VerifyRelativeError(opsin, recon, 1e-6, 1e-6);
}

TEST(DctUtilTest, Transform2x2Corners) {
  constexpr size_t N = kBlockDim;
  constexpr int block_size = N * N;
  Image3F opsin = OpsinTestImage();
  const size_t xsize_blocks = opsin.xsize() / N;
  const size_t ysize_blocks = opsin.ysize() / N;

  Image3F coeffs(xsize_blocks * block_size, ysize_blocks);
  TransposedScaledDCT(opsin, &coeffs);
  Image3F t1 = GetPixelSpaceImageFrom0HVD_64(coeffs);
  Image3F t2 = Subsample(TransposedScaledIDCT(KeepOnly2x2Corners(coeffs)), 4);
  VerifyRelativeError(t1, t2, 1e-6, 1e-6);
}

TEST(DctUtilTest, Roundtrip2x2Corners) {
  constexpr size_t N = kBlockDim;
  constexpr int block_size = N * N;
  Image3F opsin = OpsinTestImage();
  const size_t xsize_blocks = opsin.xsize() / N;
  const size_t ysize_blocks = opsin.ysize() / N;

  Image3F coeffs(xsize_blocks * block_size, ysize_blocks);
  TransposedScaledDCT(opsin, &coeffs);
  Image3F tmp = GetPixelSpaceImageFrom0HVD_64(coeffs);
  Image3F coeffs_out = CopyImage(coeffs);
  ZeroOut2x2(&coeffs_out);
  Add2x2CornersFromPixelSpaceImage(tmp, &coeffs_out);
  VerifyRelativeError(coeffs, coeffs_out, 1e-6, 1e-6);
}

using testing::FloatNear;
using testing::Pointwise;

TEST(DctUtilTest, TestRoundtripScatterGather16) {
  float input_block[4 * kDCTBlockSize];
  std::iota(input_block, input_block + 4 * kDCTBlockSize, 0.0f);

  float scattered_block[4 * kDCTBlockSize];
  ScatterBlock<16, 16>(input_block, 2 * kDCTBlockSize, scattered_block,
                       2 * kDCTBlockSize);

  float gathered_block[4 * kDCTBlockSize];
  GatherBlock<16, 16>(scattered_block, 2 * kDCTBlockSize, gathered_block,
                      2 * kDCTBlockSize);

  EXPECT_THAT(gathered_block, Pointwise(FloatNear(1e-5), input_block));
}

TEST(DctUtilTest, TestRoundtripScatterGather32) {
  float input_block[16 * kDCTBlockSize];
  std::iota(input_block, input_block + 16 * kDCTBlockSize, 0.0f);

  float scattered_block[16 * kDCTBlockSize];
  ScatterBlock<32, 32>(input_block, 4 * kDCTBlockSize, scattered_block,
                       4 * kDCTBlockSize);

  float gathered_block[16 * kDCTBlockSize];
  GatherBlock<32, 32>(scattered_block, 4 * kDCTBlockSize, gathered_block,
                      4 * kDCTBlockSize);

  EXPECT_THAT(gathered_block, Pointwise(FloatNear(1e-5), input_block));
}

}  // namespace
}  // namespace pik
