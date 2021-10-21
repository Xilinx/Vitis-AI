// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/opsin_image.h"

#include <stddef.h>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/approx_cube_root.h"
#include "pik/codec.h"
#include "pik/compiler_specific.h"
#include "pik/external_image.h"
#include "pik/profiler.h"

namespace pik {

namespace {

PIK_INLINE float SimpleGamma(float v) { return ApproxCubeRoot(v); }

void LinearXybTransform(float r, float g, float b, float* PIK_RESTRICT valx,
                        float* PIK_RESTRICT valy, float* PIK_RESTRICT valz) {
  *valx = (kScaleR * r - kScaleG * g) * 0.5f;
  *valy = (kScaleR * r + kScaleG * g) * 0.5f;
  *valz = b;
}

}  // namespace

void LinearToXyb(const float r, const float g, const float b,
                 float* PIK_RESTRICT valx, float* PIK_RESTRICT valy,
                 float* PIK_RESTRICT valz) {
  float mixed[3];
  OpsinAbsorbance(r, g, b, mixed);
  for (size_t c = 0; c < 3; ++c) {
    // mixed should be non-negative even for wide-gamut. Make sure of that:
    mixed[c] = std::max(0.0f, mixed[c]);
    mixed[c] = SimpleGamma(mixed[c]);
  }
  LinearXybTransform(mixed[0], mixed[1], mixed[2], valx, valy, valz);

  // For wide-gamut inputs, r/g/b and valx (but not y/z) are often negative.
}

// This is different from butteraugli::OpsinDynamicsImage() in the sense that
// it does not contain a sensitivity multiplier based on the blurred image.
Image3F OpsinDynamicsImage(const CodecInOut* in, const Rect& in_rect) {
  PROFILER_FUNC;

  // Convert to linear sRGB (unless already in that space)
  const Image3F* linear_srgb = &in->color();
  Image3F copy;
  Rect linear_rect = in_rect;
  if (!in->IsLinearSRGB()) {
    const ColorEncoding& c = in->Context()->c_linear_srgb[in->IsGray()];
    PIK_CHECK(in->CopyTo(in_rect, c, &copy));
    linear_srgb = &copy;
    // We've cut out the rectangle, start at x0=y0=0 in copy.
    linear_rect = Rect(copy);
  }

  const size_t xsize = in_rect.xsize();
  const size_t ysize = in_rect.ysize();
  Image3F opsin(xsize, ysize);

  for (size_t y = 0; y < ysize; ++y) {
    const float* PIK_RESTRICT row_in0 =
        linear_rect.ConstPlaneRow(*linear_srgb, 0, y);
    const float* PIK_RESTRICT row_in1 =
        linear_rect.ConstPlaneRow(*linear_srgb, 1, y);
    const float* PIK_RESTRICT row_in2 =
        linear_rect.ConstPlaneRow(*linear_srgb, 2, y);
    float* PIK_RESTRICT row_xyb0 = opsin.PlaneRow(0, y);
    float* PIK_RESTRICT row_xyb1 = opsin.PlaneRow(1, y);
    float* PIK_RESTRICT row_xyb2 = opsin.PlaneRow(2, y);
    for (size_t x = 0; x < xsize; x++) {
      LinearToXyb(row_in0[x], row_in1[x], row_in2[x], &row_xyb0[x],
                  &row_xyb1[x], &row_xyb2[x]);
    }
  }
  return opsin;
}

// DEPRECATED
Image3F OpsinDynamicsImage(const Image3B& srgb8) {
  CodecContext codec_context;
  CodecInOut io(&codec_context);
  Image3F srgb = StaticCastImage3<float>(srgb8);
  io.SetFromImage(std::move(srgb), codec_context.c_srgb[0]);
  PIK_CHECK(io.TransformTo(codec_context.c_linear_srgb[io.IsGray()]));
  return OpsinDynamicsImage(&io, Rect(io.color()));
}

}  // namespace pik
