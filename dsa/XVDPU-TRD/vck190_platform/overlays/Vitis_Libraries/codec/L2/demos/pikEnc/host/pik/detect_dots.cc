// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <cstdio>
#include <string>
#include "pik/codec.h"
#include "pik/data_parallel.h"
#include "pik/file_io.h"
#include "pik/gauss_blur.h"
#include "pik/image.h"
#include "pik/image_ops.h"
#include "pik/opsin_image.h"
#include "pik/opsin_inverse.h"
#include "pik/opsin_params.h"

namespace pik {

namespace {

Image3F LoadImage(const std::string& name) {
  CodecContext codec_context;
  CodecInOut io(&codec_context);
  PIK_CHECK(io.SetFromFile(name, /*pool=*/nullptr));
  return pik::OpsinDynamicsImage(&io, Rect(io.color()));
}

void DumpImage(const Image3F& img, const char* filename) {
  Image3F linear(img.xsize(), img.ysize());
  CopyImageTo(img, &linear);
  OpsinToLinear(&linear);

  CodecContext ctx;
  CodecInOut io(&ctx);
  io.SetFromImage(std::move(linear), ctx.c_linear_srgb[0]);
  PIK_ASSERT(io.EncodeToFile(ctx.c_srgb[0], 8, filename));
}

bool IsLonely(Rect rect, const ImageF& img, float delta) {
  // At most this many neighbors with large difference in the rect.
  constexpr size_t kHowLonely = 3;
  //  A neighbor has large delta, if it is at least kHowHigh times delta.
  const double kHowHigh = 0.37559546016936962;
  size_t count = 0;
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const float* row = rect.ConstRow(img, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      if (row[x] >= kHowHigh * delta) {
        if (count++ > kHowLonely) {
          return false;
        }
      }
    }
  }
  return true;
}
}  // namespace

void SplitDots(const Image3F& image, Image3F* without_dots, Image3F* dots) {
  // Parameters for the Gaussian.
  constexpr size_t kRadius = 5;
  const double kSigma = 3.0;
  // When to consider a delta to be large enough for further investigation.
  const double kDotThreshold = 0.85;
  // Side length of the rect for the neighborhood.
  constexpr size_t kSide = 7;
  CompressParams params;
  CodecContext codec_context;
  CodecInOut output_image(&codec_context);
  ColorEncoding encoding;
  encoding.transfer_function = TransferFunction::kLinear;
  PIK_ASSERT(ColorManagement::SetProfileFromFields(&encoding));
  ImageF sum_of_squares(image.xsize(), image.ysize());
  std::vector<double> gauss_kernelD = GaussianKernel(kRadius, kSigma);
  std::vector<float> gauss_kernel(gauss_kernelD.size());
  for (int i = 0; i < gauss_kernelD.size(); ++i) {
    gauss_kernel[i] = gauss_kernelD[i];
  }
  *without_dots = Convolve(image, gauss_kernel);
  CopyImageTo(image, dots);
  SubtractFrom(*without_dots, dots);

  for (size_t y = 0; y < image.ysize(); ++y) {
    std::array<const float*, 3> rows;
    std::array<const float*, 3> dot_rows;
    float* sos_row = sum_of_squares.Row(y);
    for (size_t c = 0; c < 3; c++) {
      rows[c] = image.Plane(c).ConstRow(y);
      dot_rows[c] = dots->Plane(c).ConstRow(y);
    }
    for (size_t x = 0; x < image.xsize(); ++x) {
      sos_row[x] = 0.0f;
      for (size_t c = 0; c < 3; c++) {
        sos_row[x] += dot_rows[c][x] * dot_rows[c][x];
      }
    }
  }

  for (size_t y = 0; y < image.ysize(); ++y) {
    const float* PIK_RESTRICT sos_row = sum_of_squares.ConstRow(y);
    const float* PIK_RESTRICT rows[3];
    float* PIK_RESTRICT dots_rows[3];
    float* PIK_RESTRICT without_dots_rows[3];
    for (size_t c = 0; c < 3; c++) {
      rows[c] = image.Plane(c).ConstRow(y);
      dots_rows[c] = dots->PlaneRow(c, y);
      without_dots_rows[c] = without_dots->PlaneRow(c, y);
    }
    for (size_t x = 0; x < image.xsize(); ++x) {
      bool keep_original = true;

      if ((sos_row[x] > kDotThreshold)) {
        Rect rect = Rect(std::max<int64_t>(x - kSide / 2, 0),
                         std::max<int64_t>(y - kSide / 2, 0), kSide, kSide,
                         image.xsize(), image.ysize());
        if (IsLonely(rect, sum_of_squares, sos_row[x])) {
          std::array<float, 3> medians = Image3Median(image, rect);
          keep_original = false;
          for (size_t c = 0; c < 3; c++) {
            without_dots_rows[c][x] = medians[c];
            dots_rows[c][x] = rows[c][x] - without_dots_rows[c][x];
          }
        }
      }

      if (keep_original) {
        for (size_t c = 0; c < 3; c++) {
          without_dots_rows[c][x] = rows[c][x];
          dots_rows[c][x] = 0.0f;
        }
      }
    }
  }
}
}  // namespace pik
