// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/yuv_opsin_convert.h"

#include <stdint.h>
#include <algorithm>
#include <array>
#include <type_traits>

#include "pik/compiler_specific.h"
#include "pik/gamma_correct.h"

namespace pik {

static constexpr double kScaleR = 1.001746913108605;
static constexpr double kScaleG = 2.0 - kScaleR;
static constexpr double kInvScaleR = 1.0 / kScaleR;
static constexpr double kInvScaleG = 1.0 / kScaleG;

static constexpr double kOpsinAbsorbanceMatrix[9] = {
    0.355028246972028, 0.589422218034148, 0.055549534993826,
    0.250871605395556, 0.714937756329137, 0.034190638275308,
    0.091915449087840, 0.165250230906774, 0.742834320005384,
};

static constexpr double kOpsinAbsorbanceInverseMatrix[9] = {
    6.805644286129,  -5.552270790544, -0.253373707795,
    -2.373074275591, 3.349796660147,  0.023277709773,
    -0.314192274838, -0.058176067042, 1.372368367449,
};
static constexpr double kXCenter = 0.008714601398;
static constexpr float kXRadius = 0.035065606236;

constexpr double kScaleY = 219.0 / 255.0;
constexpr double kScaleUV = 112.0 / 255.0;

constexpr double kOffsetY = 0.0625;
constexpr double kOffsetUV = 0.5;

constexpr double kScaleX = 0.3;
constexpr double kScaleB = 0.5;

#define clamp(V, M) (uint16_t)((V) < 0 ? 0 : ((V) > (M) ? (M) : V))

double SrgbToLinear(double val) {
  if (val < 0.0) return 0.0;
  if (val <= 0.04045) return val / 12.92;
  if (val >= 1.0) return 1.0;
  return std::pow((val + 0.055) / 1.055, 2.4);
}

double LinearToSrgb(double val) {
  if (val < 0.0) return 0.0;
  if (val >= 1.0) return 1.0;
  if (val <= 0.04045 / 12.92) return val * 12.92;
  return std::pow(val, 1.0 / 2.4) * 1.055 - 0.055;
}

double SimpleGamma(double x) {
  return x < 0.04 / 29.16 ? x * 29.16 : std::pow(x, 1.0 / 3.0) * 1.08 - 0.08;
}

double SimpleGammaInverse(double x) {
  return x < 0.04 ? x / 29.16 : std::pow((x + 0.08) / 1.08, 3);
}

// Input range:  [0 .. (1<<bits)-1]
// Output range: [0.0 .. 1.0]
void YUVOpsinPixelToRGB(uint16_t yv, uint16_t uv, uint16_t vv, int bits,
                        double* r, double* g, double* b) {
  const double norm = 1. / ((1 << bits) - 1);
  const double Y = yv * norm;
  const double U = uv * norm;
  const double V = vv * norm;

  const double valy = (Y - kOffsetY) / kScaleY;
  const double valb = (U - kOffsetUV) / kScaleUV / kScaleB;
  const double valx = (V - kOffsetUV) * kXRadius / kScaleUV / kScaleX;

  const double bmg = valb + valy;
  const double rmg = kInvScaleR * (valx + kXCenter + valy);
  const double gmg = kInvScaleG * (valy - valx - kXCenter);

  const double rm = SimpleGammaInverse(rmg);
  const double gm = SimpleGammaInverse(gmg);
  const double bm = SimpleGammaInverse(bmg);

  const double rl = (kOpsinAbsorbanceInverseMatrix[0] * rm +
                     kOpsinAbsorbanceInverseMatrix[1] * gm +
                     kOpsinAbsorbanceInverseMatrix[2] * bm);
  const double gl = (kOpsinAbsorbanceInverseMatrix[3] * rm +
                     kOpsinAbsorbanceInverseMatrix[4] * gm +
                     kOpsinAbsorbanceInverseMatrix[5] * bm);
  const double bl = (kOpsinAbsorbanceInverseMatrix[6] * rm +
                     kOpsinAbsorbanceInverseMatrix[7] * gm +
                     kOpsinAbsorbanceInverseMatrix[8] * bm);

  *r = LinearToSrgb(rl);
  *g = LinearToSrgb(gl);
  *b = LinearToSrgb(bl);
#if YUV_OPSIN_DEBUG
  printf("y: %d  u: %d  v: %d\n", yv, uv, vv);
  printf("Y: %.10f  U: %.10f  V: %.10f\n", Y, U, V);
  printf("valx: %.10f  valy: %.10f  valb: %.10f\n", valx, valy, valb);
  printf("rmg: %.10f  gmg: %.10f  bmg: %.10f\n", rmg, gmg, bmg);
  printf("rm: %.10f  gm: %.10f  bm: %.10f\n", rm, gm, bm);
  printf("rl: %.10f  gl: %.10f  bl: %.10f\n", rl, gl, bl);
  printf("r: %.10f  g: %.10f  b: %.10f\n", *r, *g, *b);
#endif
}

// Input range:  [0 .. (1<<bits)-1]
template <typename T>
void YUVOpsinPixelToRGB(uint16_t yv, uint16_t uv, uint16_t vv, int bits, T* r,
                        T* g, T* b) {
  const int maxv_out = (1 << (8 * sizeof(T))) - 1;
  double rd, gd, bd;
  YUVOpsinPixelToRGB(yv, uv, vv, bits, &rd, &gd, &bd);
  *r = clamp(.5 + maxv_out * rd, maxv_out);
  *g = clamp(.5 + maxv_out * gd, maxv_out);
  *b = clamp(.5 + maxv_out * bd, maxv_out);
#if YUV_OPSIN_DEBUG
  printf("r: %d  g: %d  b: %d\n", *r, *g, *b);
#endif
}

// Input range:  [0.0 .. 1.0]
// Output range: [0 .. (1<<bits)-1]
void RGBPixelToYUVOpsin(double r, double g, double b, int bits, uint16_t* y,
                        uint16_t* u, uint16_t* v) {
  const double rl = SrgbToLinear(r);
  const double gl = SrgbToLinear(g);
  const double bl = SrgbToLinear(b);

  const double rm =
      (kOpsinAbsorbanceMatrix[0] * rl + kOpsinAbsorbanceMatrix[1] * gl +
       kOpsinAbsorbanceMatrix[2] * bl);
  const double gm =
      (kOpsinAbsorbanceMatrix[3] * rl + kOpsinAbsorbanceMatrix[4] * gl +
       kOpsinAbsorbanceMatrix[5] * bl);
  const double bm =
      (kOpsinAbsorbanceMatrix[6] * rl + kOpsinAbsorbanceMatrix[7] * gl +
       kOpsinAbsorbanceMatrix[8] * bl);

  const double rmg = SimpleGamma(rm);
  const double gmg = SimpleGamma(gm);
  const double bmg = SimpleGamma(bm);

  const double valx = (kScaleR * rmg - kScaleG * gmg) * 0.5 - kXCenter;
  const double valy = (kScaleR * rmg + kScaleG * gmg) * 0.5;
  const double valb = (bmg - valy);

  const double Y = kOffsetY + kScaleY * valy;
  const double U = kOffsetUV + kScaleUV * kScaleB * valb;
  const double V = kOffsetUV + kScaleUV * kScaleX * valx / kXRadius;

  const double maxv = (1 << bits) - 1;
  *y = clamp(.5 + maxv * Y, maxv);
  *u = clamp(.5 + maxv * U, maxv);
  *v = clamp(.5 + maxv * V, maxv);
#if YUV_OPSIN_DEBUG
  printf("rl: %.10f  gl: %.10f  bl: %.10f\n", rl, gl, bl);
  printf("rm: %.10f  gm: %.10f  bm: %.10f\n", rm, gm, bm);
  printf("rmg: %.10f  gmg: %.10f  bmg: %.10f\n", rmg, gmg, bmg);
  printf("valx: %.10f  valy: %.10f  valb: %.10f\n", valx, valy, valb);
  printf("Y: %.10f  U: %.10f  V: %.10f\n", Y, U, V);
  printf("y: %d  u: %d  v: %d\n", *y, *u, *v);
#endif
}

// Output range: [0 .. (1<<bits)-1]
template <typename T>
void RGBPixelToYUVOpsin(T r, T g, T b, int bits, uint16_t* y, uint16_t* u,
                        uint16_t* v) {
  const double norm = 1. / ((1 << (8 * sizeof(T))) - 1);
  const double rd = r * norm;
  const double gd = g * norm;
  const double bd = b * norm;
  RGBPixelToYUVOpsin(rd, gd, bd, bits, y, u, v);
#if YUV_OPSIN_DEBUG
  printf("r: %d  g: %d  b: %d\n", r, g, b);
  printf("r: %.10f  g: %.10f  b: %.10f\n", rd, gd, bd);
#endif
}

//
// Wrapper functions to convert between 8-bit, 16-bit or linear sRGB images
// and 8, 10 or 12 bit YUV Opsin images.
//

template <typename T>
void YUVOpsinImageToRGB(const Image3U& yuv, int bit_depth, Image3<T>* rgb) {
  for (size_t y = 0; y < yuv.ysize(); ++y) {
    const uint16_t* PIK_RESTRICT row_yuv0 = yuv.PlaneRow(0, y);
    const uint16_t* PIK_RESTRICT row_yuv1 = yuv.PlaneRow(1, y);
    const uint16_t* PIK_RESTRICT row_yuv2 = yuv.PlaneRow(2, y);
    T* PIK_RESTRICT row_rgb0 = rgb->PlaneRow(0, y);
    T* PIK_RESTRICT row_rgb1 = rgb->PlaneRow(1, y);
    T* PIK_RESTRICT row_rgb2 = rgb->PlaneRow(2, y);
    for (size_t x = 0; x < yuv.xsize(); ++x) {
      YUVOpsinPixelToRGB(row_yuv0[x], row_yuv1[x], row_yuv2[x], bit_depth,
                         &row_rgb0[x], &row_rgb1[x], &row_rgb2[x]);
    }
  }
}

Image3B RGB8ImageFromYUVOpsin(const Image3U& yuv, int bit_depth) {
  Image3B rgb(yuv.xsize(), yuv.ysize());
  YUVOpsinImageToRGB(yuv, bit_depth, &rgb);
  return rgb;
}

Image3U RGB16ImageFromYUVOpsin(const Image3U& yuv, int bit_depth) {
  Image3U rgb(yuv.xsize(), yuv.ysize());
  YUVOpsinImageToRGB(yuv, bit_depth, &rgb);
  return rgb;
}

Image3F RGBLinearImageFromYUVOpsin(const Image3U& yuv, int bit_depth) {
  Image3F rgb(yuv.xsize(), yuv.ysize());
  for (size_t y = 0; y < yuv.ysize(); ++y) {
    const uint16_t* PIK_RESTRICT row_yuv0 = yuv.PlaneRow(0, y);
    const uint16_t* PIK_RESTRICT row_yuv1 = yuv.PlaneRow(1, y);
    const uint16_t* PIK_RESTRICT row_yuv2 = yuv.PlaneRow(2, y);
    float* PIK_RESTRICT row_linear0 = rgb.PlaneRow(0, y);
    float* PIK_RESTRICT row_linear1 = rgb.PlaneRow(1, y);
    float* PIK_RESTRICT row_linear2 = rgb.PlaneRow(2, y);
    for (size_t x = 0; x < yuv.xsize(); ++x) {
      double rd, gd, bd;
      YUVOpsinPixelToRGB(row_yuv0[x], row_yuv1[x], row_yuv2[x], bit_depth, &rd,
                         &gd, &bd);
      row_linear0[x] = Srgb8ToLinearDirect(rd * 255.0);
      row_linear1[x] = Srgb8ToLinearDirect(gd * 255.0);
      row_linear2[x] = Srgb8ToLinearDirect(bd * 255.0);
    }
  }
  return rgb;
}

template <typename T>
void RGBImageToYUVOpsin(const Image3<T>& rgb, int bit_depth, Image3U* yuv) {
  for (size_t y = 0; y < rgb.ysize(); ++y) {
    const T* PIK_RESTRICT row_rgb0 = rgb.ConstPlaneRow(0, y);
    const T* PIK_RESTRICT row_rgb1 = rgb.ConstPlaneRow(1, y);
    const T* PIK_RESTRICT row_rgb2 = rgb.ConstPlaneRow(2, y);
    uint16_t* PIK_RESTRICT row_yuv0 = yuv->PlaneRow(0, y);
    uint16_t* PIK_RESTRICT row_yuv1 = yuv->PlaneRow(1, y);
    uint16_t* PIK_RESTRICT row_yuv2 = yuv->PlaneRow(2, y);
    for (size_t x = 0; x < rgb.xsize(); ++x) {
      RGBPixelToYUVOpsin(row_rgb0[x], row_rgb1[x], row_rgb2[x], bit_depth,
                         &row_yuv0[x], &row_yuv1[x], &row_yuv2[x]);
    }
  }
}

Image3U YUVOpsinImageFromRGB8(const Image3B& rgb, int out_bit_depth) {
  Image3U yuv(rgb.xsize(), rgb.ysize());
  RGBImageToYUVOpsin(rgb, out_bit_depth, &yuv);
  return yuv;
}

Image3U YUVOpsinImageFromRGB16(const Image3U& rgb, int out_bit_depth) {
  Image3U yuv(rgb.xsize(), rgb.ysize());
  RGBImageToYUVOpsin(rgb, out_bit_depth, &yuv);
  return yuv;
}

Image3U YUVOpsinImageFromRGBLinear(const Image3F& rgb, int out_bit_depth) {
  Image3U yuv(rgb.xsize(), rgb.ysize());
  const double norm = 1. / 255.;
  for (size_t y = 0; y < yuv.ysize(); ++y) {
    const float* PIK_RESTRICT row_linear0 = rgb.ConstPlaneRow(0, y);
    const float* PIK_RESTRICT row_linear1 = rgb.ConstPlaneRow(1, y);
    const float* PIK_RESTRICT row_linear2 = rgb.ConstPlaneRow(2, y);
    uint16_t* PIK_RESTRICT row_yuv0 = yuv.PlaneRow(0, y);
    uint16_t* PIK_RESTRICT row_yuv1 = yuv.PlaneRow(1, y);
    uint16_t* PIK_RESTRICT row_yuv2 = yuv.PlaneRow(2, y);

    for (size_t x = 0; x < yuv.xsize(); ++x) {
      double rd = LinearToSrgb8Direct(row_linear0[x]) * norm;
      double gd = LinearToSrgb8Direct(row_linear1[x]) * norm;
      double bd = LinearToSrgb8Direct(row_linear2[x]) * norm;
      RGBPixelToYUVOpsin(rd, gd, bd, out_bit_depth, &row_yuv0[x], &row_yuv1[x],
                         &row_yuv2[x]);
    }
  }
  return yuv;
}

}  // namespace pik
