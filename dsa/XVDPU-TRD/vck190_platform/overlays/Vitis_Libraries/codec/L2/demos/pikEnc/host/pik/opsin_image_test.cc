// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/opsin_image.h"

#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <array>

#include "gtest/gtest.h"
#include "pik/codec.h"
#include "pik/color_encoding.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/linalg.h"
#include "pik/opsin_params.h"

namespace pik {
namespace {

TEST(OpsinImageTest, VerifyOpsinAbsorbanceInverseMatrix) {
  float matrix[9];  // writable copy
  for (int i = 0; i < 9; i++) {
    matrix[i] = GetOpsinAbsorbanceInverseMatrix()[i];
  }
  Inv3x3Matrix(matrix);
  for (int i = 0; i < 9; i++) {
    EXPECT_NEAR(matrix[i], kOpsinAbsorbanceMatrix[i], 1e-6);
  }
}

// Compute min/max based on extremal linear sRGB values.
TEST(OpsinImageTest, XybRange) {
  printf("Current XYB minmax constants: %.4f %.4f  %.4f %.4f  %.4f %.4f\n",
         kXybMin[0], kXybMax[0], kXybMin[1], kXybMax[1], kXybMin[2],
         kXybMax[2]);

  Image3F linear(1u << 16, 257);
  for (int b = 0; b < 256; ++b) {
    float* PIK_RESTRICT row0 = linear.PlaneRow(0, b + 1);
    float* PIK_RESTRICT row1 = linear.PlaneRow(1, b + 1);
    float* PIK_RESTRICT row2 = linear.PlaneRow(2, b + 1);
    for (int r = 0; r < 256; ++r) {
      for (int g = 0; g < 256; ++g) {
        const int x = (r << 8) + g;
        row0[x] = r;
        row1[x] = g;
        row2[x] = b;
      }
    }
  }
  CodecContext codec_context;
  CodecInOut io(&codec_context);
  io.SetFromImage(std::move(linear), codec_context.c_linear_srgb[0]);
  Image3F opsin = OpsinDynamicsImage(
      &io, Rect(0, 1, io.color().xsize(), io.color().ysize() - 1));
  for (int c = 0; c < 3; ++c) {
    float minval = 1e10f;
    float maxval = -1e10f;
    int rgb_min = 0;
    int rgb_max = 0;
    for (int b = 0; b < 256; ++b) {
      const float* PIK_RESTRICT row = opsin.PlaneRow(c, b);
      for (int r = 0; r < 256; ++r) {
        for (int g = 0; g < 256; ++g) {
          float val = row[(r << 8) + g];
          if (val < minval) {
            minval = val;
            rgb_min = (r << 16) + (g << 8) + b;
          }
          if (val > maxval) {
            maxval = val;
            rgb_max = (r << 16) + (g << 8) + b;
          }
        }
      }
    }
    printf(
        "Opsin image plane %d range: [%8.4f, %8.4f] "
        "center: %.12f, range: %.12f (RGBmin=%06x, RGBmax=%06x)\n",
        c, minval, maxval, 0.5 * (minval + maxval), 0.5 * (maxval - minval),
        rgb_min, rgb_max);
    // Ensure our constants are at least as wide as those obtained from sRGB.
    EXPECT_LE(kXybMin[c], minval) << "in plane " << c;
    EXPECT_GE(kXybMax[c], maxval) << "in plane " << c;
  }
}

// NOTE: opsin_inverse contains round-trip test.

}  // namespace
}  // namespace pik
