// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/bilinear_transform.h"
#include "pik/common.h"
#include "pik/image.h"

constexpr bool kUseBilinearTransforms = false;

std::tuple<double, double> ForwardCoordTransform(
    double x_out, double y_out, double edge,
    const double *transform_source_coords) {
  // (0.5, 0.5) <- (source[0], source[1])
  // (edge - 0.5, 0.5) <- (source[2], source[3])
  // (edge - 0.5, edge - 0.5) <- (source[4], source[5])
  // (0.5, edge - 0.5) <- (source[6], source[7])

  const double x_prop = (x_out - 0.5) / (edge - 1.0);
  const double y_prop = (y_out - 0.5) / (edge - 1.0);

  const double x = transform_source_coords[0] * (1 - x_prop) * (1 - y_prop) +
                   transform_source_coords[2] * x_prop * (1 - y_prop) +
                   transform_source_coords[4] * x_prop * y_prop +
                   transform_source_coords[6] * (1 - x_prop) * y_prop;

  const double y = transform_source_coords[1] * (1 - x_prop) * (1 - y_prop) +
                   transform_source_coords[3] * x_prop * (1 - y_prop) +
                   transform_source_coords[5] * x_prop * y_prop +
                   transform_source_coords[7] * (1 - x_prop) * y_prop;

  return std::tie(x, y);
}

std::tuple<double, double> ReverseCoordTransform(
    double x_in, double y_in, double edge,
    const double *transform_source_coords) {
  double cur_guess_x = 0.5, cur_guess_y = 0.5;
  constexpr int kNewtonIters = 10;

  for (int i = 0; i < kNewtonIters; i++) {
    const double x0 = cur_guess_x, x1 = 1 - cur_guess_x;
    const double y0 = cur_guess_y, y1 = 1 - cur_guess_y;

    const double guess_out_x = transform_source_coords[0] * x1 * y1 +
                               transform_source_coords[2] * x0 * y1 +
                               transform_source_coords[4] * x0 * y0 +
                               transform_source_coords[6] * x1 * y0;
    const double guess_out_y = transform_source_coords[1] * x1 * y1 +
                               transform_source_coords[3] * x0 * y1 +
                               transform_source_coords[5] * x0 * y0 +
                               transform_source_coords[7] * x1 * y0;
    const double j00 =
        -transform_source_coords[0] * y1 + transform_source_coords[2] * y1 +
        transform_source_coords[4] * y0 - transform_source_coords[6] * y0;
    const double j01 =
        -transform_source_coords[1] * y1 + transform_source_coords[3] * y1 +
        transform_source_coords[5] * y0 - transform_source_coords[7] * y0;
    const double j10 =
        -transform_source_coords[0] * x1 - transform_source_coords[2] * x0 +
        transform_source_coords[4] * x0 + transform_source_coords[6] * x1;
    const double j11 =
        -transform_source_coords[1] * x1 - transform_source_coords[3] * x0 +
        transform_source_coords[5] * x0 + transform_source_coords[7] * x1;

    const double inv_det_j = 1 / (j00 * j11 - j01 * j10);

    const double res_x = guess_out_x - x_in;
    const double res_y = guess_out_y - y_in;

    cur_guess_x -= inv_det_j * (j11 * res_x - j10 * res_y);
    cur_guess_y -= inv_det_j * (-j01 * res_x + j00 * res_y);

    cur_guess_x = std::max(std::min(cur_guess_x, 1.0), 0.0);
    cur_guess_y = std::max(std::min(cur_guess_y, 1.0), 0.0);
  }

  cur_guess_x = cur_guess_x * (edge - 1.0) + 0.5;
  cur_guess_y = cur_guess_y * (edge - 1.0) + 0.5;

  return std::tie(cur_guess_x, cur_guess_y);
}

// This function implements a standard approximation to a bicubic interpolation
// via a convolution, originally from this paper:
// doi:10.1109/tassp.1981.1163711, parametrized by a = -0.5. Which after mild
// algebra results in the following polynomial to be evaluated with Horner:
// 2 f0+(f1-fn1) t+(-5 f0+4 f1-f2+2 fn1) t^2+(3 f0-3 f1+f2-fn1) t^3
double CubicInterp(double t, double fn1, double f0, double f1, double f2) {
  double h = 3 * f0 - 3 * f1 + f2 - fn1;
  h = t * h + (-5 * f0 + 4 * f1 - f2 + 2 * fn1);
  h = t * h + (f1 - fn1);
  h = t * h + 2 * f0;
  return h * 0.5;
}

namespace pik {

double CubicInterpAtCoords(size_t f_x, size_t f_y, double p_x, double p_y,
                           const ImageF &in_img, const Rect &tile_rect) {
  constexpr int kNumCubicSamplePoints = 4;
  double int_vx[kNumCubicSamplePoints];

  for (int i = 0; i < kNumCubicSamplePoints; i++) {
    const float *cur_row = tile_rect.ConstRow(in_img, f_y - 1 + i);
    int_vx[i] = CubicInterp(p_x, cur_row[f_x - 1], cur_row[f_x],
                            cur_row[f_x + 1], cur_row[f_x + 2]);
  }

  return CubicInterp(p_y, int_vx[0], int_vx[1], int_vx[2], int_vx[3]);
}

enum class InterpType { kNN, kBilinear, kBicubic };

double DeterminePixValue(double x, double y, const ImageF &in_img,
                         const Rect &tile_rect, InterpType type) {
  size_t f_x = static_cast<size_t>(std::floor(x));
  size_t f_y = static_cast<size_t>(std::floor(y));

  if (f_x < 0 || f_x >= kTileDim || f_y < 0 || f_y >= kTileDim) {
    return 0;
  }

  if (type == InterpType::kNN) {
    return tile_rect.ConstRow(in_img, f_y)[f_x];
  }

  f_x = static_cast<size_t>(std::floor(x - 0.5));
  f_y = static_cast<size_t>(std::floor(y - 0.5));
  const size_t c_x = f_x + 1;
  const size_t c_y = f_y + 1;
  const double p_x = x - f_x - 0.5;
  const double p_y = y - f_y - 0.5;

  if (!(x >= 0.5 && c_x < kTileDim && y >= 0.5 && c_y < kTileDim))
    return DeterminePixValue(x, y, in_img, tile_rect, InterpType::kNN);

  const float *fy_row = tile_rect.ConstRow(in_img, f_y);
  const float *cy_row = tile_rect.ConstRow(in_img, c_y);

  if (type == InterpType::kBilinear) {
    const double v_ff = fy_row[f_x];
    const double v_fc = cy_row[f_x];
    const double v_cf = fy_row[c_x];
    const double v_cc = cy_row[c_x];

    return (1 - p_x) * (1 - p_y) * v_ff + p_x * (1 - p_y) * v_cf +
           (1 - p_x) * p_y * v_fc + p_x * p_y * v_cc;
  }

  if (!(f_x >= 1 && c_x + 1 < kTileDim && f_y >= 1 && c_y + 1 < kTileDim))
    return DeterminePixValue(x, y, in_img, tile_rect, InterpType::kBilinear);

  return CubicInterpAtCoords(f_x, f_y, p_x, p_y, in_img, tile_rect);
}

const double kTransformInputs[8] = {0.5,
                                    kTileDim * 0.5,
                                    kTileDim * 0.5,
                                    0.5,
                                    kTileDim - 0.5,
                                    kTileDim * 0.25,
                                    kTileDim * 0.5,
                                    kTileDim - 0.5};

// TODO(user): Separate parameter selection to keep consistency.
BilinearParams ApplyReverseBilinear(Image3F *opsin) {
  const size_t xtiles = opsin->xsize() / kTileDim;
  const size_t ytiles = opsin->ysize() / kTileDim;

  if (!kUseBilinearTransforms) return BilinearParams(xtiles, ytiles);

  ImageF new_tile(kTileDim, kTileDim);

  for (size_t c = 0; c < 3; c++) {
    const ImageF &in_plane = opsin->Plane(c);

    for (size_t yt = 0; yt < ytiles; yt++) {
      for (size_t xt = 0; xt < xtiles; xt++) {
        size_t xbase = xt * kTileDim;
        size_t ybase = yt * kTileDim;
        Rect tile_rect(xbase, ybase, kTileDim, kTileDim);

        for (size_t y = 0; y < kTileDim; y++) {
          float *out_row = new_tile.Row(y);
          for (size_t x = 0; x < kTileDim; x++) {
            const double c_x = x + 0.5;
            const double c_y = y + 0.5;

            double rev_x, rev_y;
            std::tie(rev_x, rev_y) =
                ReverseCoordTransform(c_x, c_y, kTileDim, kTransformInputs);

            out_row[x] = DeterminePixValue(rev_x, rev_y, in_plane, tile_rect,
                                           InterpType::kBicubic);
          }
        }

        for (size_t y = 0; y < kTileDim; y++) {
          float *out_row = tile_rect.PlaneRow(opsin, c, y);
          const float *in_row = new_tile.ConstRow(y);
          for (size_t x = 0; x < kTileDim; x++) {
            out_row[x] = in_row[x];
          }
        }
      }
    }
  }

  return BilinearParams(xtiles, ytiles);
}

void ApplyForwardBilinear(Image3F *opsin, size_t downsample) {
  if (!kUseBilinearTransforms) return;

  PIK_ASSERT(downsample == 1);

  ImageF new_tile(kTileDim, kTileDim);

  const size_t xtiles = opsin->xsize() / kTileDim;
  const size_t ytiles = opsin->ysize() / kTileDim;

  for (size_t c = 0; c < 3; c++) {
    const ImageF &in_plane = opsin->Plane(c);
    for (size_t yt = 0; yt < ytiles; yt++) {
      for (size_t xt = 0; xt < xtiles; xt++) {
        size_t xbase = xt * kTileDim;
        size_t ybase = yt * kTileDim;
        Rect tile_rect(xbase, ybase, kTileDim, kTileDim);

        for (size_t y = 0; y < kTileDim; y++) {
          float *out_row = new_tile.Row(y);
          for (size_t x = 0; x < kTileDim; x++) {
            const double c_x = x + 0.5;
            const double c_y = y + 0.5;

            double src_x, src_y;
            std::tie(src_x, src_y) =
                ForwardCoordTransform(c_x, c_y, kTileDim, kTransformInputs);

            out_row[x] = DeterminePixValue(src_x, src_y, in_plane, tile_rect,
                                           InterpType::kBicubic);
          }
        }

        for (size_t y = 0; y < kTileDim; y++) {
          float *out_row = tile_rect.PlaneRow(opsin, c, y);
          const float *in_row = new_tile.ConstRow(y);
          for (size_t x = 0; x < kTileDim; x++) {
            out_row[x] = in_row[x];
          }
        }
      }
    }
  }
}

}  // namespace pik
