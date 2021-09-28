// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/gradient_map.h"

#include "pik/bit_reader.h"
#include "pik/entropy_coder.h"
#include "pik/image.h"
#include "pik/opsin_params.h"
#include "pik/padded_bytes.h"

namespace pik {
namespace {

// Size of the superblock, in amount of DCT blocks. So we operate on
// blocks of kNumBlocks_ * kNumBlocks_ DC components, or 8x8 times as much
// original image pixels.
const size_t kNumBlocks = 8;

double Interpolate(double v00, double v01, double v10, double v11, double x,
                   double y) {
  return v00 * (1 - x) * (1 - y) + v10 * x * (1 - y) + v01 * (1 - x) * y +
         v11 * x * y;
}

// Computes the max of the horizontal and vertical second derivative for each
// pixel, where second derivative means absolute value of difference of left
// delta and right delta (top/bottom for vertical direction).
// The radius over which the derivative is computed is only 1 pixel and it only
// checks two angles (hor and ver), but this approximation works well enough.
ImageF Gradient2(const Image3F& image, const size_t c) {
  size_t xsize = image.xsize();
  size_t ysize = image.ysize();
  ImageF image2(image.xsize(), image.ysize());
  for (size_t y = 1; y + 1 < ysize; y++) {
    const float* PIK_RESTRICT row0 = image.PlaneRow(c, y - 1);
    const float* PIK_RESTRICT row1 = image.PlaneRow(c, y);
    const float* PIK_RESTRICT row2 = image.PlaneRow(c, y + 1);
    float* row_out = image2.Row(y);
    for (int x = 1; x + 1 < xsize; x++) {
      float ddx = (row1[x] - row1[x - 1]) - (row1[x + 1] - row1[x]);
      float ddy = (row1[x] - row0[x]) - (row2[x] - row1[x]);
      row_out[x] = std::max(fabsf(ddx), fabsf(ddy));
    }
  }
  // Copy to the borders
  if (ysize > 2) {
    float* PIK_RESTRICT row0 = image2.Row(0);
    const float* PIK_RESTRICT row1 = image2.Row(1);
    const float* PIK_RESTRICT row2 = image2.Row(ysize - 2);
    float* PIK_RESTRICT row3 = image2.Row(ysize - 1);
    for (size_t x = 1; x + 1 < xsize; x++) {
      row0[x] = row1[x];
      row3[x] = row2[x];
    }
  } else {
    const float* row0_in = image.PlaneRow(c, 0);
    const float* row1_in = image.PlaneRow(c, ysize - 1);
    float* row0_out = image2.Row(0);
    float* row1_out = image2.Row(ysize - 1);
    for (size_t x = 1; x + 1 < xsize; x++) {
      // Image too narrow, take first derivative instead
      row0_out[x] = row1_out[x] = fabsf(row0_in[x] - row1_in[x]);
    }
  }
  if (xsize > 2) {
    for (size_t y = 0; y < ysize; y++) {
      float* row = image2.Row(y);
      row[0] = row[1];
      row[xsize - 1] = row[xsize - 2];
    }
  } else {
    for (size_t y = 0; y < ysize; y++) {
      const float* PIK_RESTRICT row_in = image.PlaneRow(c, y);
      float* row_out = image2.Row(y);
      // Image too narrow, take first derivative instead
      row_out[0] = row_out[xsize - 1] = fabsf(row_in[0] - row_in[xsize - 1]);
    }
  }
  return image2;
}

// Grows or shrinks binary image. Negative r makes it erode. Modifies the
// image in-place, integer is used because it goes out of boolean range for
// intermediate values.
void DilateImage(std::vector<int>& image, size_t w, size_t h, int r) {
  bool erode = false;
  if (r < 0) {
    erode = true;
    r = -r;
  }
  // First pass: distance from top to bottom and left to right.
  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      size_t i = y * w + x;
      if (!image[i] == erode) {
        image[i] = 0;
      } else {
        image[i] = w + h;
        if (y > 0) {
          image[i] = std::min<int>(image[i], image[(y - 1) * w + x] + 1);
        }
        if (x > 0) {
          image[i] = std::min<int>(image[i], image[y * w + x - 1] + 1);
        }
      }
    }
  }
  // Second pass: distance from bottom to top and right to left.
  for (int y = h - 1; y >= 0; y--) {
    for (int x = w - 1; x >= 0; x--) {
      int i = y * w + x;
      if (y + 1 < h) {
        image[i] = std::min<int>(image[i], image[(y + 1) * w + x] + 1);
      }
      if (x + 1 < w) {
        image[i] = std::min<int>(image[i], image[y * w + x + 1] + 1);
      }
    }
  }
  // Convert computed distances into new binary image.
  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      int i = y * w + x;
      image[i] = (image[i] <= r) ? !erode : erode;
    }
  }
}

std::vector<int> ThresholdImage(const ImageF& image, float v) {
  std::vector<int> result(image.xsize() * image.ysize());
  for (int y = 0; y < image.ysize(); y++) {
    const float* row = image.Row(y);
    for (int x = 0; x < image.xsize(); x++) {
      // Smaller is included, larger excluded.
      result[y * image.xsize() + x] = (row[x] > v) ? 0 : 1;
    }
  }
  return result;
}

void LinePieceFit(const float* p, size_t size, int bs, float exclude,
                  bool guess_initial, float* r) {
  size_t m = (size + bs - 2) / bs + 1;
  std::vector<int> indices(size);
  std::vector<float> included;
  size_t n = 0;
  for (int j = 0; j < size; j++) {
    if (p[j] != exclude) {
      indices[j] = n;
      included.push_back(p[j]);
      n++;
    } else {
      indices[j] = -1;
    }
  }
  // matrix F: One basis function per row, one included point per column.
  std::vector<float> f(n * m, 0);
  for (int i = 0; i < m; i++) {
    int j0 = (i - 1) * bs;
    int jm = i * bs;
    int j1 = (i + 1) * bs;

    int jbegin = std::max<int>(0, (i - 1) * bs);
    int jend = std::min<int>(size - 1, (i + 1) * bs);

    for (int j = jbegin; j <= jend; j++) {
      if (p[j] == exclude) continue;
      // This is the linear interpolation function (a triangle), to fit line
      // segments.
      float v = 0;
      if (j < jm) {
        v = (j - j0) * 1.0 / (jm - j0);
      } else if (j == jm) {
        v = 1.0;
      } else {
        v = 1.0 - (j - jm) * 1.0 / (j1 - jm);
      }
      f[i * n + indices[j]] = v;
    }
  }

  if (guess_initial) {
    // Simple heuristic: guess points that match block corners, but skip
    // excluded points.
    float prev = 0;
    for (int i = 0; i < m; i++) {
      int j = i * bs;
      r[i] = (j < size && p[j] != exclude) ? p[j] : prev;
      prev = r[i];
    }
  }

  FEM(f.data(), m, n, included.data(), r);
}

// Finds values for block corner points r that best fit the points p with
// linear interpolation.
// p must have xsize * ysize points, bs is the block size, r must have
// ((xsize + bs - 2) / bs + 1) * ((ysize + bs - 2) / bs + 1) values.
// Set point values to the value of exclude to not take them into account,
// and enable guess_initial to let this function guess initial values, false
// to use user-chosen initial values of r.
void PlanePieceFit(const float* p, size_t xsize, size_t ysize, int bs,
                   float exclude, bool guess_initial, float* r) {
  // Size of result r
  size_t xsize2 = ((xsize + bs - 2) / bs + 1);
  size_t ysize2 = ((ysize + bs - 2) / bs + 1);

  // Done with separate horizontal and vertical pass rather than globally
  // to avoid large memory and CPU uage.

  // Temporary buffer between the two passes
  std::vector<float> t(xsize2 * ysize);

  // Horizontal pass
  for (size_t y = 0; y < ysize; y++) {
    LinePieceFit(&p[y * xsize], xsize, bs, exclude, guess_initial,
                 &t[y * xsize2]);
  }

  // Vertical pass
  std::vector<float> t2(ysize);
  std::vector<float> r2(ysize2);
  for (size_t x = 0; x < xsize2; x++) {
    for (size_t y = 0; y < ysize; y++) {
      t2[y] = t[y * xsize2 + x];
    }
    LinePieceFit(&t2[0], ysize, bs, exclude, guess_initial, &r2[0]);
    for (size_t y = 0; y < ysize2; y++) {
      r[y * xsize2 + x] = r2[y];
    }
  }
}

// Computes the smooth gradient image from the computed corner points.
Image3F ComputeGradientImage(const GradientMap& gradient) {
  Image3F upscaled(gradient.xsize_dc, gradient.ysize_dc);
  for (size_t by = 0; by < gradient.ysize - 1; ++by) {
    for (int c = 0; c < 3; c++) {
      const float* row0 = gradient.gradient.PlaneRow(c, by);
      const float* row1 = gradient.gradient.PlaneRow(c, by + 1);
      for (size_t bx = 0; bx + 1 < gradient.xsize; bx++) {
        float v00 = row0[bx];
        float v01 = row1[bx];
        float v10 = row0[bx + 1];
        float v11 = row1[bx + 1];
        // x1 and y1 are exclusive endpoints and are valid coordinates
        // because there is one more point than amount of blocks.
        size_t x0 = bx * kNumBlocks;
        size_t x1 = x0 + kNumBlocks;
        size_t xend = std::min<size_t>(gradient.xsize_dc - 1, x0 + kNumBlocks);
        size_t y0 = by * kNumBlocks;
        size_t y1 = y0 + kNumBlocks;
        size_t yend = std::min<size_t>(gradient.ysize_dc - 1, y0 + kNumBlocks);
        float dx = x1 - x0;
        float dy = y1 - y0;
        for (size_t y = y0; y <= yend; y++) {
          float* row_out = upscaled.PlaneRow(c, y);
          for (size_t x = x0; x <= xend; x++) {
            row_out[x] =
                Interpolate(v00, v01, v10, v11, (x - x0) / dx, (y - y0) / dy);
          }
        }
      }
    }
  }
  return upscaled;
}

Image3S Quantize(const GradientMap& gradient, const Rect& map_rect,
                 const Quantizer& quantizer) {
  const Image3F& image = gradient.gradient;
  Image3S out(map_rect.xsize(), map_rect.ysize());
  for (int c = 0; c < 3; c++) {
    // Skip x and b channels if grayscale, but do initialize them to 0.
    if (gradient.grayscale && c != 1) {
      for (size_t y = 0; y < out.ysize(); y++) {
        int16_t* PIK_RESTRICT row_out = out.PlaneRow(c, y);
        for (size_t x = 0; x < out.xsize(); x++) {
          row_out[x] = 0;
        }
      }
      continue;
    };
    const float step = quantizer.inv_quant_dc() *
                       quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0];
    float range = kXybRadius[c] * 2;
    // Use around 3x more bits than DC's quantization, capped
    int steps = std::min(std::max(16, (int)(3 * range / step)), 255);
    float mul = steps / range;

    for (size_t y = 0; y < map_rect.ysize(); y++) {
      const float* PIK_RESTRICT row = map_rect.ConstPlaneRow(image, c, y);
      int16_t* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      const uint8_t* PIK_RESTRICT apply_row =
          gradient.apply.ConstPlaneRow(c, y);
      for (size_t x = 0; x < map_rect.xsize(); x++) {
        int value = std::round((row[x] - kXybMin[c]) * mul);
        if (apply_row[x]) {
          value = std::min(std::max(0, value), steps - 1) + 1;
        } else {
          value = 0;
        }
        row_out[x] = value;
      }
    }
  }
  return out;
}

void Dequantize(const Quantizer& quantizer, const Image3S& quant,
                GradientMap* gradient) {
  gradient->gradient = Image3F(gradient->xsize, gradient->ysize);
  for (int c = 0; c < 3; c++) {
    if (gradient->grayscale && c != 1) continue;
    const float step = quantizer.inv_quant_dc() *
                       quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0];
    float range = kXybRadius[c] * 2;
    // Use around 3x more bits than DC's quantization, capped
    int steps = std::min(std::max(16, (int)(3 * range / step)), 255);
    float mul = range / steps;

    for (size_t y = 0; y < gradient->ysize; y++) {
      float* PIK_RESTRICT row_out = gradient->gradient.PlaneRow(c, y);
      const int16_t* PIK_RESTRICT row = quant.PlaneRow(c, y);
      uint8_t* PIK_RESTRICT row_apply = gradient->apply.PlaneRow(c, y);
      for (size_t x = 0; x < gradient->xsize; x++) {
        float v;
        if (row[x] != 0) {
          v = (row[x] - 1) * mul + kXybMin[c];
          row_apply[x] = true;
        } else {
          v = 0;
          row_apply[x] = false;
        }
        row_out[x] = v;
      }
    }
  }
}

void InitGradientMap(size_t xsize_dc, size_t ysize_dc, bool grayscale,
                     GradientMap* gradient) {
  gradient->xsize_dc = xsize_dc;
  gradient->ysize_dc = ysize_dc;
  gradient->grayscale = grayscale;

  // numx and numy are amount of blocks in x and y direction, and the
  // amount is such that when there are N * kNumBlocks + 1 pixels, there
  // are only N blocks (the one extra pixel can still be part of the last
  // block), once there are N * kNumBlocks + 2 pixels, there are N + 1
  // blocks. Note that kNumBlocks is in fact the size of 1 block, num blocks
  // refers to amount of DC values (from DCT blocks) this block contains.
  size_t numx = DivCeil(xsize_dc - 1, kNumBlocks);
  size_t numy = DivCeil(ysize_dc - 1, kNumBlocks);

  // Size of the gradient map: one bigger than numx and numy because the
  // blocks have values on all corners ("fenceposts").
  gradient->xsize = numx + 1;
  gradient->ysize = numy + 1;

  // Note that the gradient is much smaller than the DC image, and the DC image
  // in turn already is much smaller than the full original image.
  gradient->gradient = Image3F(gradient->xsize, gradient->ysize);
  gradient->apply = Image3B(gradient->xsize, gradient->ysize);
}

// Serializes and deserializes the gradient image so it has the values the
// decoder will see.
void AccountForQuantization(const Quantizer& quantizer, GradientMap* gradient) {
  Image3S quantized = Quantize(
      *gradient, Rect(0, 0, gradient->xsize, gradient->ysize), quantizer);
  Dequantize(quantizer, quantized, gradient);
}
}  // namespace

// Computes the gradient map for the given image of DC values.
void ComputeGradientMap(const Image3F& opsin, bool grayscale,
                        const Quantizer& quantizer,
                        GradientMap* gradient) {
  InitGradientMap(opsin.xsize(), opsin.ysize(), grayscale, gradient);
  size_t xsize_dc = gradient->xsize_dc;
  size_t ysize_dc = gradient->ysize_dc;
  size_t xsize = gradient->xsize;
  size_t ysize = gradient->ysize;

  gradient->gradient = Image3F(xsize, ysize);
  for(int task = 0; task < 3; ++task) {
    static const float kExclude = 999999;
    static const float kMaxDiff[3] = {0.001, 0.01, 0.05};
    const size_t c = task;
    if (grayscale && c != 1) return;
    std::vector<float> points(ysize_dc * xsize_dc, kExclude);
    ImageF gradient2 = Gradient2(opsin, c);
    std::vector<int> apply = ThresholdImage(gradient2, kMaxDiff[c]);
    DilateImage(apply, xsize_dc, ysize_dc, -8);
    DilateImage(apply, xsize_dc, ysize_dc, 8);

    for (size_t by = 0; by + 1 < ysize; by++) {
      for (size_t bx = 0; bx + 1 < xsize; bx++) {
        size_t x0 = bx * kNumBlocks;
        size_t x1 = std::min<size_t>(xsize_dc, x0 + kNumBlocks);
        size_t y0 = by * kNumBlocks;
        size_t y1 = std::min<size_t>(ysize_dc, y0 + kNumBlocks);
        // Block is one larger than normal if on a right or bottom edge
        // with particular size.
        if (bx + 2 == xsize && xsize_dc % kNumBlocks == 1) x1++;
        if (by + 2 == ysize && ysize_dc % kNumBlocks == 1) y1++;
        size_t dx = x1 - x0;
        size_t dy = y1 - y0;

        for (size_t sy = 0; sy < dy; sy++) {
          for (size_t sx = 0; sx < dx; sx++) {
            int x = x0 + sx;
            int y = y0 + sy;
            if (apply[y * xsize_dc + x]) {
              points[y * xsize_dc + x] = opsin.PlaneRow(c, y)[x];
            }
          }
        }
      }
    }

    const float mul =
        1.0f / (quantizer.inv_quant_dc() *
                quantizer.DequantMatrix(0, kQuantKindDCT8, task)[0]);
    std::vector<float> coeffs(xsize * ysize);
    PlanePieceFit(points.data(), xsize_dc, ysize_dc, kNumBlocks, kExclude, true,
                  coeffs.data());
    for (size_t y = 0; y < ysize; ++y) {
      float* PIK_RESTRICT row = gradient->gradient.PlaneRow(c, y);
      const float* PIK_RESTRICT packed_row = &coeffs[y * xsize];
      uint8_t* PIK_RESTRICT apply_row = gradient->apply.PlaneRow(c, y);
      memcpy(row, packed_row, xsize * sizeof(float));
      for (size_t x = 0; x < xsize; ++x) {
        // TODO(lode): figure out when the gradient map is not needed in a
        // proper way.
        if (std::abs(3.0f * row[x] * mul) > 0.5f) {
          apply_row[x] = 1;
        } else {
          apply_row[x] = 0;
        }
      }
    }
  }

  AccountForQuantization(quantizer, gradient);
}

// Applies the stored gradient map in the decoder.
void ApplyGradientMap(const GradientMap& gradient, const Quantizer& quantizer,
                      Image3F* opsin) {
  Image3F upscaled = ComputeGradientImage(gradient);
  size_t xsize_dc = gradient.xsize_dc;
  size_t ysize_dc = gradient.ysize_dc;
  static const float kScale[3] = {1.0, 1.0, 1.0};

  for (int c = 0; c < 3; ++c) {
    if (gradient.grayscale && c != 1) return;
    const float step = quantizer.inv_quant_dc() *
                       quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0] *
                       kScale[c];

    std::vector<int> apply(gradient.ysize_dc * gradient.xsize_dc, 0);

    for (size_t y = 0; y < ysize_dc; y++) {
      float* PIK_RESTRICT row_out = opsin->PlaneRow(c, y);
      const float* PIK_RESTRICT row_in = upscaled.ConstPlaneRow(c, y);
      const uint8_t* PIK_RESTRICT row_apply =
          gradient.apply.ConstPlaneRow(c, y / kNumBlocks);
      for (size_t x = 0; x < xsize_dc; x++) {
        float diff = fabs(row_out[x] - row_in[x]);
        if (diff < step && row_apply[x / kNumBlocks]) {
          apply[y * xsize_dc + x] = 1;
        }
      }
    }

    // Reduce the size of the field where to apply the gradient, to avoid
    // doing it in noisy areas
    DilateImage(apply, gradient.xsize_dc, gradient.ysize_dc, -3);

    for (size_t y = 0; y < ysize_dc; y++) {
      float* PIK_RESTRICT row_out = opsin->PlaneRow(c, y);
      const float* PIK_RESTRICT row_in = upscaled.ConstPlaneRow(c, y);
      for (size_t x = 0; x < xsize_dc; x++) {
        if (apply[y * xsize_dc + x]) {
          row_out[x] = row_in[x];
        }
      }
    }
  }
}

void SerializeGradientMap(const GradientMap& gradient, const Rect& rect,
                          const Quantizer& quantizer, PaddedBytes* compressed) {
  PIK_ASSERT(rect.x0() % kNumBlocks == 0);
  PIK_ASSERT(rect.y0() % kNumBlocks == 0);
  Rect map_rect(rect.x0() / kNumBlocks, rect.y0() / kNumBlocks,
                DivCeil(rect.xsize() - 1, kNumBlocks) + 1,
                DivCeil(rect.ysize() - 1, kNumBlocks) + 1);
  Image3S quantized = Quantize(gradient, map_rect, quantizer);
  Image3S residuals(map_rect.xsize(), map_rect.ysize());
  ShrinkDC(map_rect, quantized, &residuals);
  std::string encoded = EncodeImageData(Rect(residuals), residuals, nullptr);
  size_t pos = compressed->size();
  compressed->resize(compressed->size() + encoded.size());
  for (size_t i = 0; i < encoded.size(); i++) {
    compressed->data()[pos++] = encoded[i];
  }
}

Status DeserializeGradientMap(size_t xsize_dc, size_t ysize_dc, bool grayscale,
                              const Quantizer& quantizer,
                              const PaddedBytes& compressed, size_t* byte_pos,
                              GradientMap* gradient) {
  InitGradientMap(xsize_dc, ysize_dc, grayscale, gradient);

  BitReader reader(compressed.data() + *byte_pos,
                   compressed.size() - *byte_pos);

  ImageS gmap_y_tmp(gradient->xsize, gradient->ysize);
  ImageS gmap_xz_res_tmp(gradient->xsize * 2, gradient->ysize);
  ImageS gmap_xz_exp_tmp(gradient->xsize * 2, gradient->ysize);

  Image3S gmap_quant(gradient->xsize, gradient->ysize);

  if (!DecodeImage(&reader, Rect(gmap_quant), &gmap_quant)) {
    return PIK_FAILURE("Failed to decode gradient map");
  }

  *byte_pos += reader.Position();

  ExpandDC(Rect(gmap_quant), &gmap_quant, &gmap_y_tmp, &gmap_xz_res_tmp,
           &gmap_xz_exp_tmp);

  Dequantize(quantizer, gmap_quant, gradient);

  return true;  // success
}

}  // namespace pik
