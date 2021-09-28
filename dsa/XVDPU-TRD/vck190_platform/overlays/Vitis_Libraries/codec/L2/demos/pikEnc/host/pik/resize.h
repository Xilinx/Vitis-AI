// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_RESIZE_H_
#define PIK_RESIZE_H_

#include <stddef.h>

#include "pik/common.h"
#include "pik/image.h"
#include "pik/status.h"
#include "pik/upscaler.h"

namespace pik {

// Ideally, this is 1.0f, but butteraugli likes bigger values (me too),
// like 1.25 for 2x resampling and 1.5 for 4x resampling.
constexpr float kUpdateScale = 1.0f;

namespace {
class Columns {
 public:
  Columns(float* start) : start_(start) {}

  float Read(size_t i) const { return start_[i]; }

  void Write(size_t i, float value) const { start_[i] = value; }

  float* start_;
};

class EvenColumns {
 public:
  EvenColumns(float* start) : start_(start) {}

  float Read(size_t i) const { return start_[2 * i]; }

  void Write(size_t i, float value) const { start_[2 * i] = value; }

  float* start_;
};

class OddColumns {
 public:
  OddColumns(float* start) : start_(start) {}

  float Read(size_t i) const { return start_[2 * i + 1]; }

  void Write(size_t i, float value) const { start_[2 * i + 1] = value; }

  float* start_;
};

class Rows {
 public:
  Rows(float* start, size_t stride) : start_(start), stride_(stride) {}

  float Read(size_t i) const { return start_[stride_ * i]; }

  void Write(size_t i, float value) const { start_[stride_ * i] = value; }

  float* start_;
  size_t stride_;
};

class EvenRows {
 public:
  EvenRows(float* start, size_t stride) : start_(start), stride_(stride) {}

  float Read(size_t i) const { return start_[2 * i * stride_]; }

  void Write(size_t i, float value) const { start_[2 * i * stride_] = value; }

  float* start_;
  size_t stride_;
};

class OddRows {
 public:
  OddRows(float* start, size_t stride) : start_(start), stride_(stride) {}

  float Read(size_t i) const { return start_[(2 * i + 1) * stride_]; }

  void Write(size_t i, float value) const {
    start_[(2 * i + 1) * stride_] = value;
  }

  float* start_;
  size_t stride_;
};

// TODO(user): SIMDify all the row/column processing.

template <class From, class To>
void F2(size_t n, const From& from, const To& to, float* tmp) {
  PIK_ASSERT(n > 8);
  constexpr float alpha[9] = {// a = (3 - sqrt(8)); [a; -a, a^2, -a^3, ...]
                              0.1715728752538099023966225736f,
                              -0.1715728752538099023966225736f,
                              0.02943725152285941437973531699f,
                              -0.005050633883346583881789307292f,
                              0.0008665517772200889110005228959f,
                              -0.0001486767799739495842138294335f,
                              0.00002550890262360859428245360423f,
                              -0.000004376635767701981480892173763f,
                              0.0000007509119826032946028994350462f};

  constexpr float mul = 0.5857864376269049511983113385f;  // (4 * a) / (1 + a)

  float y_last = from.Read(0);
  for (size_t i = 1; i <= 8; ++i) y_last += from.Read(i - 1) * alpha[i];
  tmp[0] = y_last;
  for (size_t i = 1; i < n; ++i) {
    float y_current = from.Read(i) - alpha[0] * y_last;
    tmp[i] = y_current;
    y_last = y_current;
  }

  y_last = from.Read(n - 1);
  for (size_t i = 1; i <= 8; ++i) y_last += from.Read(n - i) * alpha[i];
  float x_last = from.Read(n - 1);
  to.Write(n - 1, mul * (tmp[n - 1] + y_last));
  for (size_t i = n - 2; i < n; --i) {
    float y_current = x_last - alpha[0] * y_last;
    x_last = from.Read(i);
    to.Write(i, mul * (tmp[i] + y_current));
    y_last = y_current;
  }
}

template <class From, class To>
void Phi2(size_t n, const From& from, const To& to, float* tmp) {
  F2(n, from, to, tmp);
  for (size_t i = n - 1; i > 0; --i) to.Write(i, 0.5f * to.Read(i - 1));
  to.Write(0, 0.5f * to.Read(0));
}

void SubsampleRow2(size_t n, float* from, float* to, float* tmp) {
  PIK_ASSERT(n > 8);
  PIK_ASSERT(n % 2 == 0);
  size_t n2 = n / 2;
  float* tmp2 = tmp + n2;
  F2(n2, EvenColumns(from), Columns(tmp2), tmp);
  for (size_t i = 0; i < n2; ++i) tmp2[i] = from[2 * i + 1] - tmp2[i];
  Phi2(n2, Columns(tmp2), Columns(tmp2), tmp);
  for (size_t i = 0; i < n2; ++i) to[i] = from[2 * i] + kUpdateScale * tmp2[i];
}

void SubsampleColumn2(size_t n, float* from, size_t from_stride, float* to,
                      size_t to_stride, float* tmp) {
  PIK_ASSERT(n > 8);
  PIK_ASSERT(n % 2 == 0);
  size_t n2 = n / 2;
  float* tmp2 = tmp + n2;
  F2(n2, EvenRows(from, from_stride), Columns(tmp2), tmp);
  for (size_t i = 0; i < n2; ++i) {
    tmp2[i] = from[from_stride * (2 * i + 1)] - tmp2[i];
  }
  Phi2(n2, Columns(tmp2), Columns(tmp2), tmp);
  for (size_t i = 0; i < n2; ++i) {
    to[i * to_stride] = from[from_stride * (2 * i)] + kUpdateScale * tmp2[i];
  }
}

void UpsampleRow2(size_t n, float* from, float* to, float* tmp) {
  PIK_ASSERT(n > 8);
  for (size_t i = n - 1; i < n; --i) {
    to[2 * i] = from[i];
  }
  F2(n, EvenColumns(to), OddColumns(to), tmp);
}

void UpsampleColumn2(size_t n, float* from, size_t from_stride, float* to,
                     size_t to_stride, float* tmp) {
  PIK_ASSERT(n > 8);
  for (size_t i = n - 1; i < n; --i) {
    to[to_stride * (2 * i)] = from[from_stride * i];
  }
  F2(n, EvenRows(to, to_stride), OddRows(to, to_stride), tmp);
}

constexpr float subL[5] = {-0.1477632789908915, 0.6043134178154527,
                           0.6017439248475215, -0.06092538825140858,
                           0.00263132457932580};

constexpr float subR[5] = {0.14335798104235847, -0.23802370655991587,
                           0.12502505370893394, 0.7550194876366351,
                           0.214621184171988};

template <class From, class To>
void Subsample32(size_t n, const From& from, const To& to) {
  PIK_ASSERT(n % 3 == 0);
  size_t n3 = n / 3;
  {
    float l = from.Read(0);
    float x0 = from.Read(0);
    float x1 = from.Read(1);
    float x2 = from.Read(2);
    float r = from.Read(3);
    to.Write(0, l * subL[0] + x0 * subL[1] + x1 * subL[2] + x2 * subL[3] +
                    r * subL[4]);
    to.Write(1, l * subR[0] + x0 * subR[1] + x1 * subR[2] + x2 * subR[3] +
                    r * subR[4]);
  }
  for (size_t i = 1; i < n3 - 1; ++i) {
    size_t f = i * 3;
    size_t t = i * 2;
    float l = from.Read(f - 1);
    float x0 = from.Read(f + 0);
    float x1 = from.Read(f + 1);
    float x2 = from.Read(f + 2);
    float r = from.Read(f + 3);
    to.Write(t + 0, l * subL[0] + x0 * subL[1] + x1 * subL[2] + x2 * subL[3] +
                        r * subL[4]);
    to.Write(t + 1, l * subR[0] + x0 * subR[1] + x1 * subR[2] + x2 * subR[3] +
                        r * subR[4]);
  }
  {
    float l = from.Read(n - 4);
    float x0 = from.Read(n - 3);
    float x1 = from.Read(n - 2);
    float x2 = from.Read(n - 1);
    float r = from.Read(n - 1);
    to.Write(2 * n3 - 2, l * subL[0] + x0 * subL[1] + x1 * subL[2] +
                             x2 * subL[3] + r * subL[4]);
    to.Write(2 * n3 - 1, l * subR[0] + x0 * subR[1] + x1 * subR[2] +
                             x2 * subR[3] + r * subR[4]);
  }
}

constexpr float upL[4] = {0.38757486500910365, 0.7620777552137453,
                          -0.22848886478020333, 0.0788362445573544};
constexpr float upC[4] = {-0.11860750280548209, 0.868540521473126,
                          0.3507502137687898, -0.100683232436434};
constexpr float upR[4] = {-0.06717404018363016, 0.13550763911335584,
                          1.0687743540167292, -0.137107952946455};

template <class From, class To>
void Upsample23(size_t n, const From& from, const To& to) {
  PIK_ASSERT(n % 2 == 0);
  size_t n2 = n / 2;
  {
    float l = from.Read(n - 3);
    float x0 = from.Read(n - 2);
    float x1 = from.Read(n - 1);
    float r = from.Read(n - 1);
    to.Write(3 * n2 - 3, l * upL[0] + x0 * upL[1] + x1 * upL[2] + r * upL[3]);
    to.Write(3 * n2 - 2, l * upC[0] + x0 * upC[1] + x1 * upC[2] + r * upC[3]);
    to.Write(3 * n2 - 1, l * upR[0] + x0 * upR[1] + x1 * upR[2] + r * upR[3]);
  }
  for (size_t i = n2 - 2; i > 0; --i) {
    size_t f = i * 2;
    size_t t = i * 3;
    float l = from.Read(f - 1);
    float x0 = from.Read(f + 0);
    float x1 = from.Read(f + 1);
    float r = from.Read(f + 2);
    to.Write(t + 0, l * upL[0] + x0 * upL[1] + x1 * upL[2] + r * upL[3]);
    to.Write(t + 1, l * upC[0] + x0 * upC[1] + x1 * upC[2] + r * upC[3]);
    to.Write(t + 2, l * upR[0] + x0 * upR[1] + x1 * upR[2] + r * upR[3]);
  }
  {
    float l = from.Read(0);
    float x0 = from.Read(0);
    float x1 = from.Read(1);
    float r = from.Read(2);
    to.Write(0, l * upL[0] + x0 * upL[1] + x1 * upL[2] + r * upL[3]);
    to.Write(1, l * upC[0] + x0 * upC[1] + x1 * upC[2] + r * upC[3]);
    to.Write(2, l * upR[0] + x0 * upR[1] + x1 * upR[2] + r * upR[3]);
  }
}

}  // namespace

static inline Image3F DownsampleImage32(Image3F& src) {
  size_t w = src.xsize();
  size_t h = src.ysize();
  PIK_ASSERT(w % 3 == 0);
  PIK_ASSERT(h % 3 == 0);

  Image3F dst((w / 3) * 2, (h / 3) * 2);

  for (int c = 0; c < 3; ++c) {
    for (size_t x = 0; x < w; ++x) {
      Subsample32(h, Rows(src.PlaneRow(c, 0) + x, src.PixelsPerRow()),
                  Rows(src.PlaneRow(c, 0) + x, src.PixelsPerRow()));
    }
  }
  h = (h / 3) * 2;

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < h; ++y) {
      Subsample32(w, Columns(src.PlaneRow(c, y)), Columns(dst.PlaneRow(c, y)));
    }
  }
  w = (w * 3) / 2;

  return dst;
}

static inline Image3F UpsampleImage23(Image3F& src, size_t orig_xsize,
                                      size_t orig_ysize) {
  PIK_ASSERT(orig_xsize % 3 == 0);
  PIK_ASSERT(orig_ysize % 3 == 0);
  size_t w = (orig_xsize / 3) * 2;
  size_t h = (orig_ysize / 3) * 2;
  PIK_ASSERT(w <= src.xsize());
  PIK_ASSERT(h <= src.ysize());

  Image3F dst((w / 2) * 3, (h / 2) * 3);

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < h; ++y) {
      Upsample23(w, Columns(src.PlaneRow(c, y)), Columns(dst.PlaneRow(c, y)));
    }
  }
  w = (w / 2) * 3;

  for (int c = 0; c < 3; ++c) {
    for (size_t x = 0; x < w; ++x) {
      Upsample23(h, Rows(dst.PlaneRow(c, 0) + x, dst.PixelsPerRow()),
                 Rows(dst.PlaneRow(c, 0) + x, dst.PixelsPerRow()));
    }
  }
  h = (h / 2) * 3;

  // dst = Blur(dst, 0.66666f);

  return dst;
}

static inline Image3F DownsampleImage2N(Image3F& src, size_t factor) {
  size_t w = src.xsize();
  size_t h = src.ysize();
  PIK_ASSERT(w % factor == 0);
  PIK_ASSERT(h % factor == 0);

  Image3F dst(w / factor, h / factor);
  std::vector<float> tmp_storage(std::max(w, h));
  float* tmp = &tmp_storage[0];

  if (factor >= 2) {
    for (int c = 0; c < 3; ++c) {
      for (size_t x = 0; x < w; ++x) {
        SubsampleColumn2(h, src.PlaneRow(c, 0) + x, src.PixelsPerRow(),
                         src.PlaneRow(c, 0) + x, src.PixelsPerRow(), tmp);
      }
    }
    h /= 2;

    if (factor >= 4) {
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < h; ++y) {
          SubsampleRow2(w, src.PlaneRow(c, y), src.PlaneRow(c, y), tmp);
        }
      }
      w /= 2;

      for (int c = 0; c < 3; ++c) {
        for (size_t x = 0; x < w; ++x) {
          SubsampleColumn2(h, src.PlaneRow(c, 0) + x, src.PixelsPerRow(),
                           src.PlaneRow(c, 0) + x, src.PixelsPerRow(), tmp);
        }
      }
      h /= 2;
    }

    for (int c = 0; c < 3; ++c) {
      for (size_t y = 0; y < h; ++y) {
        SubsampleRow2(w, src.PlaneRow(c, y), dst.PlaneRow(c, y), tmp);
      }
    }
    w /= 2;
  }

  return dst;
}

static inline Image3F UpsampleImage2N(Image3F& src, size_t factor,
                                      size_t orig_xsize, size_t orig_ysize) {
  PIK_ASSERT(orig_xsize % factor == 0);
  PIK_ASSERT(orig_ysize % factor == 0);
  size_t w = orig_xsize / factor;
  size_t h = orig_ysize / factor;
  PIK_ASSERT(w <= src.xsize());
  PIK_ASSERT(h <= src.ysize());
  Image3F dst(w * factor, h * factor);
  std::vector<float> tmp_storage(factor * std::max(w, h));
  float* tmp = &tmp_storage[0];

  if (factor >= 2) {
    for (int c = 0; c < 3; ++c) {
      for (size_t y = 0; y < h; ++y) {
        UpsampleRow2(w, src.PlaneRow(c, y), dst.PlaneRow(c, y), tmp);
      }
    }
    w *= 2;

    if (factor >= 4) {
      for (int c = 0; c < 3; ++c) {
        for (size_t x = 0; x < w; ++x) {
          UpsampleColumn2(h, dst.PlaneRow(c, 0) + x, dst.PixelsPerRow(),
                          dst.PlaneRow(c, 0) + x, dst.PixelsPerRow(), tmp);
        }
      }
      h *= 2;

      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < h; ++y) {
          UpsampleRow2(w, dst.PlaneRow(c, y), dst.PlaneRow(c, y), tmp);
        }
      }
      w *= 2;
    }

    for (int c = 0; c < 3; ++c) {
      for (size_t x = 0; x < w; ++x) {
        UpsampleColumn2(h, dst.PlaneRow(c, 0) + x, dst.PixelsPerRow(),
                        dst.PlaneRow(c, 0) + x, dst.PixelsPerRow(), tmp);
      }
    }
    h *= 2;
  }

  return dst;
}

static inline Image3F PadImage(const Image3F& in, size_t min_padding,
                               size_t factor) {
  const size_t xsize = DivCeil(in.xsize() + 2 * min_padding, factor) * factor;
  const size_t ysize = DivCeil(in.ysize() + 2 * min_padding, factor) * factor;
  const size_t left_padding = (xsize - in.xsize()) / 2;
  const size_t top_padding = (ysize - in.ysize()) / 2;
  Image3F out(xsize, ysize);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < in.ysize(); ++y) {
      const float* PIK_RESTRICT row_in = in.ConstPlaneRow(c, y);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y + top_padding);
      memcpy(row_out + left_padding, row_in, in.xsize() * sizeof(row_in[0]));
      const float firstval = row_in[0];
      for (int x = 0; x < left_padding; ++x) {
        row_out[x] = firstval;
      }
      const float lastval = row_in[in.xsize() - 1];
      for (int x = in.xsize() + left_padding; x < xsize; ++x) {
        row_out[x] = lastval;
      }
    }

    for (int y = 0; y < top_padding; ++y) {
      const float* PIK_RESTRICT row_in = out.ConstPlaneRow(c, top_padding);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
    }
    const int lastrow = in.ysize() + top_padding - 1;
    for (int y = lastrow + 1; y < ysize; ++y) {
      const float* PIK_RESTRICT row_in = out.ConstPlaneRow(c, lastrow);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
    }
  }
  return out;
}

static inline Image3F UnpadImage(const Image3F& in, size_t min_padding,
                                 size_t factor, size_t orig_xsize,
                                 size_t orig_ysize) {
  PIK_ASSERT(in.xsize() % factor == 0);
  PIK_ASSERT(in.ysize() % factor == 0);
  const size_t left_padding = (in.xsize() - orig_xsize) / 2;
  const size_t top_padding = (in.ysize() - orig_ysize) / 2;
  Image3F out(orig_xsize, orig_ysize);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < orig_ysize; ++y) {
      const float* PIK_RESTRICT row_in =
          in.ConstPlaneRow(c, y + top_padding) + left_padding;
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, orig_xsize * sizeof(row_in[0]));
    }
  }
  return out;
}

static inline uint32_t ResizePadding(size_t factor2) { return 1u; }

static inline ImageSize DownsampledImageSize(ImageSize src, size_t factor2) {
  PIK_ASSERT(factor2 == 2 || factor2 == 3 || factor2 == 4 || factor2 == 8);
  ImageSize dst;
  uint32_t min_padding = ResizePadding(factor2);
  if (factor2 == 2) {
    dst.xsize = src.xsize;
    dst.ysize = src.ysize;
  } else if (factor2 == 3) {
    dst.xsize = DivCeil(src.xsize + 2u * min_padding, 3u) * 2u;
    dst.ysize = DivCeil(src.ysize + 2u * min_padding, 3u) * 2u;
  } else {  // 4 or 8
    uint32_t factor = factor2 / 2;
    dst.xsize = DivCeil(src.xsize + 2u * min_padding, factor);
    dst.ysize = DivCeil(src.ysize + 2u * min_padding, factor);
  }
  return dst;
}

static inline Image3F DownsampleImage(Image3F& src, size_t factor2) {
  PIK_ASSERT(factor2 == 3 || factor2 == 4 || factor2 == 8);
  size_t min_padding = ResizePadding(factor2);
  size_t factor = (factor2 == 3) ? 3 : (factor2 / 2);
  Image3F padded = PadImage(src, min_padding, factor);
  return (factor2 == 3) ? DownsampleImage32(padded)
                        : DownsampleImage2N(padded, factor);
}

static inline Image3F UpsampleImage(Image3F& src, size_t orig_xsize,
                                    size_t orig_ysize, size_t factor2) {
  PIK_ASSERT(factor2 == 3 || factor2 == 4 || factor2 == 8);
  size_t factor = (factor2 == 3) ? 3 : (factor2 / 2);
  size_t min_padding = ResizePadding(factor2);
  size_t padded_xsize = DivCeil(orig_xsize + 2 * min_padding, factor) * factor;
  size_t padded_ysize = DivCeil(orig_ysize + 2 * min_padding, factor) * factor;
  Image3F upsampled =
      (factor2 == 3) ? UpsampleImage23(src, padded_xsize, padded_ysize)
                     : UpsampleImage2N(src, factor, padded_xsize, padded_ysize);
  return UnpadImage(upsampled, min_padding, factor, orig_xsize, orig_ysize);
}

}  // namespace pik

#endif  // PIK_RESIZE_H_
