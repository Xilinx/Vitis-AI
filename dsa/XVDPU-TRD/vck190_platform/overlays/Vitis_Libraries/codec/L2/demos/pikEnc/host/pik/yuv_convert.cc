// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/yuv_convert.h"

#include <stdint.h>
#include <algorithm>
#include <array>
#include <type_traits>

#include "pik/compiler_specific.h"
#include "pik/gamma_correct.h"

namespace pik {

// Conversion matrices and functions between 8 or 16 bit sRGB and
// 8, 10 or 12 bit YUV Rec 709 color spaces.

constexpr double kWeightR = 0.2126;
constexpr double kWeightB = 0.0722;
constexpr double kWeightG = 1.0 - kWeightR - kWeightB;
constexpr double kWeightBc = 1.0 - kWeightB;
constexpr double kWeightRc = 1.0 - kWeightR;
constexpr double kScaleY = 219.0 / 255.0;
constexpr double kScaleUV = 112.0 / 255.0;

// clang-format off
constexpr double RGBtoYUVMatrix[9] = {
    kWeightR * kScaleY,
    kWeightG * kScaleY,
    kWeightB * kScaleY,
    (-kWeightR / kWeightBc) * kScaleUV,
    (-kWeightG / kWeightBc) * kScaleUV,
    kScaleUV,
    kScaleUV,
    (-kWeightG / kWeightRc) * kScaleUV,
    (-kWeightB / kWeightRc) * kScaleUV,
};

constexpr double RGBtoYUVMatrixAdd[3] = {.0625, .5, .5};

constexpr double YUVtoRGBMatrix[9] = {
    1.0 / kScaleY,
    0.0,
    kWeightRc / kScaleUV,
    1.0 / kScaleY,
    -kWeightBc * kWeightB / kWeightG / kScaleUV,
    -kWeightRc * kWeightR / kWeightG / kScaleUV,
    1.0 / kScaleY,
    kWeightBc / kScaleUV,
    0.0};
// clang-format on

#define clamp(V, M) (uint16_t)((V) < 0 ? 0 : ((V) > (M) ? (M) : V))

// Input range:  [0 .. (1<<bits)-1]
// Output range: [0.0 .. 1.0]
void YUVPixelToRGB(uint16_t yv, uint16_t uv, uint16_t vv, int bits, double* r,
                   double* g, double* b) {
  const double norm = 1. / ((1 << bits) - 1);
  const double y = yv * norm - RGBtoYUVMatrixAdd[0];
  const double u = uv * norm - RGBtoYUVMatrixAdd[1];
  const double v = vv * norm - RGBtoYUVMatrixAdd[2];
  *r = YUVtoRGBMatrix[0] * y + YUVtoRGBMatrix[1] * u + YUVtoRGBMatrix[2] * v;
  *g = YUVtoRGBMatrix[3] * y + YUVtoRGBMatrix[4] * u + YUVtoRGBMatrix[5] * v;
  *b = YUVtoRGBMatrix[6] * y + YUVtoRGBMatrix[7] * u + YUVtoRGBMatrix[8] * v;
}

// Input range:  [0 .. (1<<bits)-1]
template <typename T>
void YUVPixelToRGB(uint16_t yv, uint16_t uv, uint16_t vv, int bits, T* r, T* g,
                   T* b) {
  const int maxv_out = (1 << (8 * sizeof(T))) - 1;
  double rd, gd, bd;
  YUVPixelToRGB(yv, uv, vv, bits, &rd, &gd, &bd);
  *r = clamp(.5 + maxv_out * rd, maxv_out);
  *g = clamp(.5 + maxv_out * gd, maxv_out);
  *b = clamp(.5 + maxv_out * bd, maxv_out);
}

// Input range:  [0.0 .. 1.0]
// Output range: [0 .. (1<<bits)-1]
void RGBPixelToYUV(double r, double g, double b, int bits, uint16_t* y,
                   uint16_t* u, uint16_t* v) {
  const double maxv = (1 << bits) - 1;
  const double Y = RGBtoYUVMatrixAdd[0] + RGBtoYUVMatrix[0] * r +
                   RGBtoYUVMatrix[1] * g + RGBtoYUVMatrix[2] * b;
  const double U = RGBtoYUVMatrixAdd[1] + RGBtoYUVMatrix[3] * r +
                   RGBtoYUVMatrix[4] * g + RGBtoYUVMatrix[5] * b;
  const double V = RGBtoYUVMatrixAdd[2] + RGBtoYUVMatrix[6] * r +
                   RGBtoYUVMatrix[7] * g + RGBtoYUVMatrix[8] * b;
  *y = clamp(.5 + maxv * Y, maxv);
  *u = clamp(.5 + maxv * U, maxv);
  *v = clamp(.5 + maxv * V, maxv);
}

// Output range: [0 .. (1<<bits)-1]
template <typename T>
void RGBPixelToYUV(T r, T g, T b, int bits, uint16_t* y, uint16_t* u,
                   uint16_t* v) {
  const double norm = 1. / ((1 << (8 * sizeof(T))) - 1);
  const double rd = r * norm;
  const double gd = g * norm;
  const double bd = b * norm;
  RGBPixelToYUV(rd, gd, bd, bits, y, u, v);
}

//
// Wrapper functions to convert between 8-bit, 16-bit or linear sRGB images
// and 8, 10 or 12 bit YUV Rec 709 images.
//

template <typename T>
void YUVRec709ImageToRGB(const Image3U& yuv, int bit_depth, Image3<T>* rgb) {
  for (size_t y = 0; y < yuv.ysize(); ++y) {
    const uint16_t* PIK_RESTRICT row_yuv0 = yuv.PlaneRow(0, y);
    const uint16_t* PIK_RESTRICT row_yuv1 = yuv.PlaneRow(1, y);
    const uint16_t* PIK_RESTRICT row_yuv2 = yuv.PlaneRow(2, y);

    T* PIK_RESTRICT row_rgb0 = rgb->PlaneRow(0, y);
    T* PIK_RESTRICT row_rgb1 = rgb->PlaneRow(1, y);
    T* PIK_RESTRICT row_rgb2 = rgb->PlaneRow(2, y);
    for (size_t x = 0; x < yuv.xsize(); ++x) {
      YUVPixelToRGB(row_yuv0[x], row_yuv1[x], row_yuv2[x], bit_depth,
                    &row_rgb0[x], &row_rgb1[x], &row_rgb2[x]);
    }
  }
}

Image3B RGB8ImageFromYUVRec709(const Image3U& yuv, int bit_depth) {
  Image3B rgb(yuv.xsize(), yuv.ysize());
  YUVRec709ImageToRGB(yuv, bit_depth, &rgb);
  return rgb;
}

Image3U RGB16ImageFromYUVRec709(const Image3U& yuv, int bit_depth) {
  Image3U rgb(yuv.xsize(), yuv.ysize());
  YUVRec709ImageToRGB(yuv, bit_depth, &rgb);
  return rgb;
}

Image3F RGBLinearImageFromYUVRec709(const Image3U& yuv, int bit_depth) {
  Image3F rgb(yuv.xsize(), yuv.ysize());
  for (int y = 0; y < yuv.ysize(); ++y) {
    const uint16_t* PIK_RESTRICT row_yuv0 = yuv.ConstPlaneRow(0, y);
    const uint16_t* PIK_RESTRICT row_yuv1 = yuv.ConstPlaneRow(1, y);
    const uint16_t* PIK_RESTRICT row_yuv2 = yuv.ConstPlaneRow(2, y);
    float* PIK_RESTRICT row_linear0 = rgb.PlaneRow(0, y);
    float* PIK_RESTRICT row_linear1 = rgb.PlaneRow(1, y);
    float* PIK_RESTRICT row_linear2 = rgb.PlaneRow(2, y);
    for (int x = 0; x < yuv.xsize(); ++x) {
      double rd, gd, bd;
      YUVPixelToRGB(row_yuv0[x], row_yuv1[x], row_yuv2[x], bit_depth, &rd, &gd,
                    &bd);
      row_linear0[x] = Srgb8ToLinearDirect(rd * 255.0);
      row_linear1[x] = Srgb8ToLinearDirect(gd * 255.0);
      row_linear2[x] = Srgb8ToLinearDirect(bd * 255.0);
    }
  }
  return rgb;
}

template <typename T>
void RGBImageToYUVRec709(const Image3<T>& rgb, int bit_depth, Image3U* yuv) {
  for (int y = 0; y < rgb.ysize(); ++y) {
    const T* PIK_RESTRICT row_rgb0 = rgb.ConstPlaneRow(0, y);
    const T* PIK_RESTRICT row_rgb1 = rgb.ConstPlaneRow(1, y);
    const T* PIK_RESTRICT row_rgb2 = rgb.ConstPlaneRow(2, y);
    uint16_t* PIK_RESTRICT row_yuv0 = yuv->PlaneRow(0, y);
    uint16_t* PIK_RESTRICT row_yuv1 = yuv->PlaneRow(1, y);
    uint16_t* PIK_RESTRICT row_yuv2 = yuv->PlaneRow(2, y);
    for (int x = 0; x < rgb.xsize(); ++x) {
      RGBPixelToYUV(row_rgb0[x], row_rgb1[x], row_rgb2[x], bit_depth,
                    &row_yuv0[x], &row_yuv1[x], &row_yuv2[x]);
    }
  }
}

Image3U YUVRec709ImageFromRGB8(const Image3B& rgb, int out_bit_depth) {
  Image3U yuv(rgb.xsize(), rgb.ysize());
  RGBImageToYUVRec709(rgb, out_bit_depth, &yuv);
  return yuv;
}

Image3U YUVRec709ImageFromRGB16(const Image3U& rgb, int out_bit_depth) {
  Image3U yuv(rgb.xsize(), rgb.ysize());
  RGBImageToYUVRec709(rgb, out_bit_depth, &yuv);
  return yuv;
}

Image3U YUVRec709ImageFromRGBLinear(const Image3F& rgb, int out_bit_depth) {
  Image3U yuv(rgb.xsize(), rgb.ysize());
  const double norm = 1. / 255.;
  for (int y = 0; y < yuv.ysize(); ++y) {
    const float* PIK_RESTRICT row_linear0 = rgb.ConstPlaneRow(0, y);
    const float* PIK_RESTRICT row_linear1 = rgb.ConstPlaneRow(1, y);
    const float* PIK_RESTRICT row_linear2 = rgb.ConstPlaneRow(2, y);
    uint16_t* PIK_RESTRICT row_yuv0 = yuv.PlaneRow(0, y);
    uint16_t* PIK_RESTRICT row_yuv1 = yuv.PlaneRow(1, y);
    uint16_t* PIK_RESTRICT row_yuv2 = yuv.PlaneRow(2, y);
    for (int x = 0; x < yuv.xsize(); ++x) {
      double rd = LinearToSrgb8Direct(row_linear0[x]) * norm;
      double gd = LinearToSrgb8Direct(row_linear1[x]) * norm;
      double bd = LinearToSrgb8Direct(row_linear2[x]) * norm;
      RGBPixelToYUV(rd, gd, bd, out_bit_depth, &row_yuv0[x], &row_yuv1[x],
                    &row_yuv2[x]);
    }
  }
  return yuv;
}

void SubSampleChroma(const Image3U& yuv, int bit_depth, ImageU* yplane,
                     ImageU* uplane, ImageU* vplane) {
  const int xsize = yuv.xsize();
  const int ysize = yuv.ysize();
  const int c_xsize = (xsize + 1) / 2;
  const int c_ysize = (ysize + 1) / 2;
  *yplane = CopyImage(yuv.Plane(0));
  *uplane = ImageU(c_xsize, c_ysize);
  *vplane = ImageU(c_xsize, c_ysize);
  for (int y = 0; y < c_ysize; ++y) {
    for (int x = 0; x < c_xsize; ++x) {
      int sum_u = 0;
      int sum_v = 0;
      for (int iy = 0; iy < 2; ++iy) {
        for (int ix = 0; ix < 2; ++ix) {
          int yy = std::min(2 * y + iy, ysize - 1);
          int xx = std::min(2 * x + ix, xsize - 1);
          sum_u += yuv.PlaneRow(1, yy)[xx];
          sum_v += yuv.PlaneRow(2, yy)[xx];
        }
      }
      uplane->Row(y)[x] = (sum_u + 2) / 4;
      vplane->Row(y)[x] = (sum_v + 2) / 4;
    }
  }
}

ImageU SuperSamplePlane(const ImageU& in, int bit_depth, int out_xsize,
                        int out_ysize) {
  const int c_xsize = in.xsize();
  const int c_ysize = in.ysize();
  ImageU out(2 * c_xsize, 2 * c_ysize);
  for (int y = 0; y < c_ysize; ++y) {
    const int y0 = y > 0 ? y - 1 : y;
    const int y1 = y;
    const int y2 = y + 1 < c_ysize ? y + 1 : y;
    const uint16_t* const PIK_RESTRICT row0 = in.Row(y0);
    const uint16_t* const PIK_RESTRICT row1 = in.Row(y1);
    const uint16_t* const PIK_RESTRICT row2 = in.Row(y2);
    uint16_t* const PIK_RESTRICT row_out0 = out.Row(2 * y);
    uint16_t* const PIK_RESTRICT row_out1 = out.Row(2 * y + 1);
    for (int x = 0; x < c_xsize; ++x) {
      const int x0 = x > 0 ? x - 1 : x;
      const int x1 = x;
      const int x2 = x + 1 < c_xsize ? x + 1 : x;
      row_out0[2 * x + 0] =
          (9 * row1[x1] + 3 * row1[x0] + 3 * row0[x1] + 1 * row0[x0] + 8) / 16;
      row_out0[2 * x + 1] =
          (9 * row1[x1] + 3 * row1[x2] + 3 * row0[x1] + 1 * row0[x2] + 8) / 16;
      row_out1[2 * x + 0] =
          (9 * row1[x1] + 3 * row1[x0] + 3 * row2[x1] + 1 * row2[x0] + 8) / 16;
      row_out1[2 * x + 1] =
          (9 * row1[x1] + 3 * row1[x2] + 3 * row2[x1] + 1 * row2[x2] + 8) / 16;
    }
  }
  out.ShrinkTo(out_xsize, out_ysize);
  return out;
}

Image3U SuperSampleChroma(const ImageU& yplane, const ImageU& uplane,
                          const ImageU& vplane, int bit_depth) {
  const int xsize = yplane.xsize();
  const int ysize = yplane.ysize();
  return Image3U(CopyImage(yplane),
                 SuperSamplePlane(uplane, bit_depth, xsize, ysize),
                 SuperSamplePlane(vplane, bit_depth, xsize, ysize));
}

}  // namespace pik
