// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/upscaler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>
#include "pik/compiler_specific.h"
#include "pik/image.h"
#include "pik/resample.h"

namespace pik {

namespace {

std::vector<float> ComputeKernel(float sigma) {
  // Filtering becomes slower, but more Gaussian when m is increased.
  // More Gaussian doesn't mean necessarily better results altogether.
  const float m = 2.5;
  const float scaler = -1.0 / (2 * sigma * sigma);
  const int diff = std::max<int>(1, m * fabs(sigma));
  std::vector<float> kernel(2 * diff + 1);
  for (int i = -diff; i <= diff; ++i) {
    kernel[i + diff] = exp(scaler * i * i);
  }
  return kernel;
}

void ConvolveBorderColumn(const ImageF& in, const std::vector<float>& kernel,
                          const float weight_no_border,
                          const float border_ratio, const size_t x,
                          float* const PIK_RESTRICT row_out) {
  const int offset = kernel.size() / 2;
  int minx = x < offset ? 0 : x - offset;
  int maxx = std::min<int>(in.xsize() - 1, x + offset);
  float weight = 0.0f;
  for (int j = minx; j <= maxx; ++j) {
    weight += kernel[j - x + offset];
  }
  // Interpolate linearly between the no-border scaling and border scaling.
  weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
  float scale = 1.0f / weight;
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = in.Row(y);
    float sum = 0.0f;
    for (int j = minx; j <= maxx; ++j) {
      sum += row_in[j] * kernel[j - x + offset];
    }
    row_out[y] = sum * scale;
  }
}

// Computes a horizontal convolution and transposes the result.
ImageF Convolution(const ImageF& in, const std::vector<float>& kernel,
                   const float border_ratio) {
  ImageF out(in.ysize(), in.xsize());
  const int len = kernel.size();
  const int offset = kernel.size() / 2;
  float weight_no_border = 0.0f;
  for (int j = 0; j < len; ++j) {
    weight_no_border += kernel[j];
  }
  float scale_no_border = 1.0f / weight_no_border;
  const int border1 = in.xsize() <= offset ? in.xsize() : offset;
  const int border2 = in.xsize() - offset;
  int x = 0;
  // left border
  for (; x < border1; ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out.Row(x));
  }
  // middle
  for (; x < border2; ++x) {
    float* const PIK_RESTRICT row_out = out.Row(x);
    for (size_t y = 0; y < in.ysize(); ++y) {
      const float* const PIK_RESTRICT row_in = &in.Row(y)[x - offset];
      float sum = 0.0f;
      for (int j = 0; j < len; ++j) {
        sum += row_in[j] * kernel[j];
      }
      row_out[y] = sum * scale_no_border;
    }
  }
  // right border
  for (; x < in.xsize(); ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out.Row(x));
  }
  return out;
}

// A blur somewhat similar to a 2D Gaussian blur.
// See: https://en.wikipedia.org/wiki/Gaussian_blur
ImageF Blur(const ImageF& in, float sigma, float border_ratio) {
  std::vector<float> kernel = ComputeKernel(sigma);
  return Convolution(Convolution(in, kernel, border_ratio), kernel,
                     border_ratio);
}

}  // namespace

Image3F Blur(const Image3F& image, float sigma) {
  float border = 0.0;
  return Image3F(Blur(image.Plane(0), sigma, border),
                 Blur(image.Plane(1), sigma, border),
                 Blur(image.Plane(2), sigma, border));
}

}  // namespace pik
