// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/block_dictionary.h"
#include <sys/types.h>
#include <cstdint>
#include <limits>
#include <vector>
#include "pik/codec.h"
#include "pik/detect_dots.h"
#include "pik/entropy_coder.h"
#include "pik/image.h"
#include "pik/opsin_inverse.h"
#include "pik/robust_statistics.h"
#include "pik/status.h"

namespace pik {
constexpr float kBlockScale = 3.5;
constexpr float kBlockInvScale = 1.0f / kBlockScale;
constexpr int kMaxBlocks = 1 << 24;

// TODO(veluca): choose some reasonable set of blocks as a static dictionary.
constexpr int kNumStaticBlocks = 1;
QuantizedBlock kStaticBlocks[kNumStaticBlocks] = {
    {1, 3, {{0, 0, 0}, {-11, -8, -11}, {-10, -7, -10}}},
};

enum Contexts {
  kNumBlockContext = 0,
  kBlockSizeContext = 1,
  kPixelsContextStart = 2,
  kPixelsContextY = 3,
  kPixelsContextB = 4,
  kBlockOffsetContext = 5,
  kBlockWidthContext = 6,
  kBlockIdCountContext = 7,
  kBlockIdSkipContext = 8,
  kNumBlockDictionaryContexts,
};

// We can represent numbers up to 2**kMaxBlockDictionarySymbol. As the biggest
// numbers have an image-size range, 2**32 should be more than enough here.
constexpr int kMaxBlockDictionarySymbol = 32;

float ScaleForQuantization(float val, size_t c) {
  return kBlockScale * val / kXybRadius[c];
}
int Quantize(float val, size_t c) {
  return std::round(ScaleForQuantization(val, c));
}

// BlockInfo contains the data describing a block before quantization
// and also contains its position.
struct BlockInfo {
  Rect rect;
  const Image3F* image;
  BlockInfo(const Image3F* image, Rect rect) : rect(rect), image(image) {}

  explicit operator QuantizedBlock() {
    QuantizedBlock info;
    info.xsize = rect.xsize();
    info.ysize = rect.ysize();

    for (size_t c = 0; c < 3; c++) {
      for (size_t iy = 0; iy < rect.ysize(); iy++) {
        const float* PIK_RESTRICT row = image->ConstPlaneRow(c, rect.y0() + iy);
        for (size_t ix = 0; ix < rect.xsize(); ix++) {
          info.pixels[c][iy * rect.xsize() + ix] =
              Quantize(row[rect.x0() + ix], c);
        }
      }
    }
    return info;
  }
  bool IsSimilar(const BlockInfo& other) const {
    if (other.rect.xsize() != rect.xsize()) return false;
    if (other.rect.ysize() != rect.ysize()) return false;
    float sum = 0;
    for (size_t c = 0; c < 3; c++) {
      for (size_t iy = 0; iy < rect.ysize(); iy++) {
        const float* PIK_RESTRICT row = image->ConstPlaneRow(c, rect.y0() + iy);
        const float* PIK_RESTRICT other_row =
            other.image->ConstPlaneRow(c, other.rect.y0() + iy);
        for (size_t ix = 0; ix < rect.xsize(); ix++) {
          float diff = row[rect.x0() + ix] - other_row[other.rect.x0() + ix];
          sum += diff * diff;
        }
      }
    }
    if (sum > kDotDistThreshold) return false;

    return true;
  }
};

BlockDictionary::BlockDictionary(const std::vector<QuantizedBlock>& dictionary,
                                 const std::vector<BlockPosition>& positions)
    : dictionary_(dictionary), positions_(positions) {
  std::sort(positions_.begin(), positions_.end(),
            [](const BlockPosition& a, const BlockPosition& b) {
              return std::make_tuple(a.transform, a.id, a.x, a.y, a.dx, a.dy,
                                     a.width) <
                     std::make_tuple(b.transform, b.id, b.x, b.y, b.dx, b.dy,
                                     b.width);
            });
}

std::string BlockDictionary::Encode(PikImageSizeInfo* info) const {
  std::vector<std::vector<Token>> tokens(1);

  auto add_num = [&](int context, size_t num) {
    int bits, nbits;
    EncodeVarLenUint(num, &nbits, &bits);
    tokens[0].emplace_back(context, nbits, nbits, bits);
  };

  add_num(kNumBlockContext, dictionary_.size());
  for (size_t i = 0; i < dictionary_.size(); i++) {
    const QuantizedBlock& info = dictionary_[i];
    add_num(kBlockSizeContext,
            i == 0 ? info.xsize
                   : PackSigned(info.xsize - dictionary_[i - 1].xsize));
    add_num(kBlockSizeContext,
            i == 0 ? info.ysize
                   : PackSigned(info.ysize - dictionary_[i - 1].ysize));
    for (size_t c = 0; c < 3; c++) {
      int ctx = kPixelsContextStart + c;
      for (size_t iy = 0; iy < info.ysize; iy++) {
        for (size_t ix = 0; ix < info.xsize; ix++) {
          int8_t val = info.pixels[c][iy * info.xsize + ix];
          int8_t pred = 0;
          if (ix != 0) {
            pred += info.pixels[c][iy * info.xsize + ix - 1];
          }
          if (iy != 0) {
            pred += info.pixels[c][(iy - 1) * info.xsize + ix];
          }
          if (ix != 0 && iy != 0) pred >>= 1;
          add_num(ctx, PackSigned(val - pred));
        }
      }
    }
  }
  for (size_t transform = 0; transform < 2; transform++) {
    int last_block = -1;
    size_t last_idx = 0;
    for (size_t id = 0; id < dictionary_.size() + kNumStaticBlocks; id++) {
      size_t idx = last_idx;
      while (idx < positions_.size() && positions_[idx].id == id &&
             positions_[idx].transform == transform) {
        idx++;
      }
      size_t num = idx - last_idx;
      if (num == 0) continue;
      if (last_block + 1 < id) {
        add_num(kBlockIdCountContext, 0);
        add_num(kBlockIdSkipContext, id - last_block - 2);
      }
      last_block = id;
      add_num(kBlockIdCountContext, num);
      for (size_t i = last_idx; i < idx; i++) {
        const BlockPosition& pos = positions_[i];
        add_num(
            kBlockOffsetContext,
            i == last_idx ? pos.x : PackSigned(pos.x - positions_[i - 1].x));
        add_num(
            kBlockOffsetContext,
            i == last_idx ? pos.y : PackSigned(pos.y - positions_[i - 1].y));
        PIK_ASSERT(pos.id == id);
        if (transform) {
          add_num(kBlockOffsetContext, PackSigned(pos.dx));
          add_num(kBlockOffsetContext, PackSigned(pos.dy));
          add_num(kBlockWidthContext, PackSigned(pos.width));
        }
      }
      last_idx = idx;
    }
    if (last_block + 1 < dictionary_.size() + kNumStaticBlocks) {
      add_num(kBlockIdCountContext, 0);
      add_num(kBlockIdSkipContext,
              dictionary_.size() + kNumStaticBlocks - 2 - last_block);
    }
  }

  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  std::string enc = BuildAndEncodeHistograms(
      kNumBlockDictionaryContexts, tokens, &codes, &context_map, nullptr);
  enc += WriteTokens(tokens[0], codes, context_map, nullptr);
  if (info) {
    info->total_size += enc.size();
  }
  return enc;
}

Status BlockDictionary::Decode(BitReader* br, size_t xsize, size_t ysize) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  PIK_RETURN_IF_ERROR(DecodeHistograms(br, kNumBlockDictionaryContexts,
                                       kMaxBlockDictionarySymbol, &code,
                                       &context_map));
  ANSSymbolReader decoder(&code);

  auto read_num = [&](int context) {
    br->FillBitBuffer();
    int s = decoder.ReadSymbol(context_map[context], br);
    int bits = br->ReadBits(s);
    return DecodeVarLenUint(s, bits);
  };

  size_t dict_size = read_num(kNumBlockContext);
  if (dict_size > kMaxBlocks) {
    return PIK_FAILURE("Too many blocks in dictionary");
  }

  dictionary_.resize(dict_size);
  for (size_t i = 0; i < dictionary_.size(); i++) {
    QuantizedBlock& info = dictionary_[i];
    info.xsize = read_num(kBlockSizeContext);
    info.ysize = read_num(kBlockSizeContext);
    if (i != 0) {
      info.xsize = UnpackSigned(info.xsize) + dictionary_[i - 1].xsize;
      info.ysize = UnpackSigned(info.ysize) + dictionary_[i - 1].ysize;
    }
    if (info.xsize > kMaxBlockSize)
      return PIK_FAILURE("Block xsize is too big: %lu", info.xsize);
    if (info.ysize > kMaxBlockSize)
      return PIK_FAILURE("Block ysize is too big: %lu", info.ysize);
    for (size_t c = 0; c < 3; c++) {
      int ctx = kPixelsContextStart + c;
      for (size_t iy = 0; iy < info.ysize; iy++) {
        for (size_t ix = 0; ix < info.xsize; ix++) {
          int8_t pred = 0;
          if (ix != 0) {
            pred += info.pixels[c][iy * info.xsize + ix - 1];
          }
          if (iy != 0) {
            pred += info.pixels[c][(iy - 1) * info.xsize + ix];
          }
          if (ix != 0 && iy != 0) pred >>= 1;
          info.pixels[c][iy * info.xsize + ix] =
              UnpackSigned(read_num(ctx)) + pred;
        }
      }
    }
  }

  for (size_t transform = 0; transform < 2; transform++) {
    size_t to_skip = 0;
    for (size_t id = 0; id < dictionary_.size() + kNumStaticBlocks; id++) {
      if (to_skip > 0) {
        to_skip--;
        continue;
      }
      size_t id_count = read_num(kBlockIdCountContext);
      if (id_count > kMaxBlocks) {
        return PIK_FAILURE("Too many blocks in dictionary");
      }
      if (id_count == 0) {
        to_skip = read_num(kBlockIdSkipContext);
        continue;
      }
      positions_.resize(positions_.size() + id_count);
      size_t id_start = positions_.size() - id_count;
      for (size_t i = id_start; i < positions_.size(); i++) {
        BlockPosition& pos = positions_[i];
        pos.x = read_num(kBlockOffsetContext);
        pos.y = read_num(kBlockOffsetContext);
        pos.id = id;
        const QuantizedBlock& info =
            pos.id < dictionary_.size()
                ? dictionary_[pos.id]
                : kStaticBlocks[pos.id - dictionary_.size()];
        if (i != id_start) {
          pos.x = UnpackSigned(pos.x) + positions_[i - 1].x;
          pos.y = UnpackSigned(pos.y) + positions_[i - 1].y;
        }
        if (pos.x + info.xsize > xsize) {
          return PIK_FAILURE("Invalid block x (id %lu): at %lu + %lu > %lu",
                             pos.id, pos.x, info.xsize, xsize);
        }
        if (pos.y + info.ysize > ysize) {
          return PIK_FAILURE("Invalid block y: at %lu + %lu > %lu", pos.y,
                             info.ysize, ysize);
        }
        pos.transform = transform;
        if (transform) {
          pos.dx = UnpackSigned(read_num(kBlockOffsetContext));
          pos.dy = UnpackSigned(read_num(kBlockOffsetContext));
          pos.width = UnpackSigned(read_num(kBlockWidthContext));
        }
      }
    }
    if (to_skip > 0) {
      return PIK_FAILURE("Invalid number of skipped block ids!");
    }
  }

  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  PIK_RETURN_IF_ERROR(br->JumpToByteBoundary());
  return true;
}

#ifdef PIK_BD_DUMP_IMAGES
void DumpImage(const Image3F& img) {
  Image3F linear(img.xsize(), img.ysize());
  CopyImageTo(img, &linear);
  OpsinToLinear(&linear, /*pool=*/nullptr);

  CodecContext ctx;
  CodecInOut io(&ctx);
  io.SetFromImage(std::move(linear), ctx.c_linear_srgb[0]);

  static size_t cnt = 0;
  std::string pos = "/tmp/dbg" + std::to_string(cnt++) + ".png";
  PIK_ASSERT(io.EncodeToFile(ctx.c_srgb[0], 8, pos));
}

void DumpImage(const ImageF& img) {
  Image3F l(img.xsize(), img.ysize());
  PIK_ASSERT(img.PixelsPerRow() == l.PixelsPerRow());
  std::fill(l.PlaneRow(0, 0),
            l.PlaneRow(0, 0) + img.PixelsPerRow() * img.ysize(), 0.0f);
  std::fill(l.PlaneRow(2, 0),
            l.PlaneRow(2, 0) + img.PixelsPerRow() * img.ysize(), 0.0f);
  float min, max;
  ImageMinMax(img, &min, &max);
  if (min == max) max = 1.0;
  for (size_t y = 0; y < img.ysize(); y++) {
    const float* PIK_RESTRICT row = img.Row(y);
    float* PIK_RESTRICT row_out = l.PlaneRow(1, y);
    for (size_t x = 0; x < img.xsize(); x++) {
      row_out[x] = (row[x] - min) / (max - min);
    }
  }
  DumpImage(l);
}
#endif

template <bool add>
void BlockDictionary::Apply(Image3F* opsin, size_t downsampling) const {
#ifdef PIK_BD_DUMP_IMAGES
  DumpImage(*opsin);
#endif
  if (downsampling != 1) {
    // TODO(veluca): downsampling not implemented yet.
    PIK_CHECK(positions_.empty());
  }
  // Blocks copied as-is.
  for (const BlockPosition& pos : positions_) {
    if (pos.transform) continue;
    size_t by = pos.y;
    size_t bx = pos.x;
    const QuantizedBlock& info =
        pos.id < dictionary_.size()
            ? dictionary_[pos.id]
            : kStaticBlocks[pos.id - dictionary_.size()];
    for (size_t c = 0; c < 3; c++) {
      for (size_t iy = 0; iy < info.ysize; iy++) {
        float* row = opsin->PlaneRow(c, by + iy) + bx;
        for (size_t ix = 0; ix < info.xsize; ix++) {
          float val = kBlockInvScale * kXybRadius[c] *
                      info.pixels[c][iy * info.xsize + ix];
          if (add) {
            row[ix] += val;
          } else {
            row[ix] -= val;
          }
        }
      }
    }
  }
  // Scaled/rotated blocks.
  float* PIK_RESTRICT rows[3] = {opsin->PlaneRow(0, 0), opsin->PlaneRow(1, 0),
                                 opsin->PlaneRow(2, 0)};
  size_t stride = opsin->PixelsPerRow();
  for (const BlockPosition& pos : positions_) {
    if (!pos.transform) continue;
    float block[3][(2 * kMaxBlockSize + 5) * (2 * kMaxBlockSize + 5)] = {};
    const QuantizedBlock& info =
        pos.id < dictionary_.size()
            ? dictionary_[pos.id]
            : kStaticBlocks[pos.id - dictionary_.size()];
    size_t xs = info.xsize;
    size_t ys = info.ysize;
    for (size_t c = 0; c < 3; c++) {
      for (int iy = 0; iy < int(2 * ys) + 1; iy++) {
        for (int ix = 0; ix < int(2 * xs) + 1; ix++) {
          float val = 0.0f;
          for (int dy = -1; dy < 1; dy++) {
            for (int dx = -1; dx < 1; dx++) {
              int sy = (iy + dy) / 2;
              if (sy < 0) sy = 0;
              if (sy >= ys) sy = ys - 1;
              int sx = (ix + dx) / 2;
              if (sx < 0) sx = 0;
              if (sx >= xs) sx = xs - 1;
              val += info.pixels[c][sy * info.xsize + sx];
            }
          }
          block[c][(iy + 2) * (2 * xs + 5) + (ix + 2)] = val * 0.25f;
        }
      }
    }
    float by00 = pos.y;
    float bx00 = pos.x;
    float bx01 = bx00 + pos.dx;
    float by01 = by00 + pos.dy;
    float xnorm = std::sqrt(pos.dx * pos.dx + pos.dy * pos.dy);
    float ynorm = pos.width * 0.5f;
    float invxnorm = 1.0f / xnorm;
    float invynorm = 1.0f / ynorm;
    float deltax = pos.dy * invxnorm * ynorm;
    float deltay = -pos.dx * invxnorm * ynorm;
    float bx11 = bx01 + deltax;
    float by11 = by01 + deltay;
    float bx10 = bx00 + deltax;
    float by10 = by00 + deltay;
    float inv_determinant = 1.0f / (pos.dx * deltay - pos.dy * deltax);
    float inverse_transform[4] = {
        xs * deltay * inv_determinant, -float(xs) * deltax * inv_determinant,
        -float(ys) * pos.dy * inv_determinant, ys * pos.dx * inv_determinant};
    int64_t min_x = std::min(std::min(bx00, bx01), std::min(bx10, bx11));
    int64_t min_y = std::min(std::min(by00, by01), std::min(by10, by11));
    int64_t max_x = std::max(std::max(bx00, bx01), std::max(bx10, bx11)) + 1;
    int64_t max_y = std::max(std::max(by00, by01), std::max(by10, by11)) + 1;
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > opsin->xsize()) max_x = opsin->xsize();
    if (max_y > opsin->ysize()) max_y = opsin->ysize();
    constexpr float kAntialiasingMargin = 0.3f;
    float margin_x = kAntialiasingMargin * invxnorm;
    float margin_y = kAntialiasingMargin * invynorm;
    for (size_t iy = min_y; iy < max_y; iy++) {
      for (size_t ix = min_x; ix < max_x; ix++) {
        float x = ix - bx00 + 0.5f;
        float y = iy - by00 + 0.5f;
        float ox = inverse_transform[0] * x + inverse_transform[1] * y;
        float oy = inverse_transform[2] * x + inverse_transform[3] * y;
        if (ox >= -margin_x && ox < xs + margin_x && oy >= -margin_y &&
            oy < ys + margin_y) {
          ox = 2 * ox + 2;
          oy = 2 * oy + 2;
          int floorx = ox;
          int ceilx = floorx + 1;
          float fracx = ox - floorx;
          int floory = oy;
          int ceily = floory + 1;
          float fracy = oy - floory;
          for (size_t c = 0; c < 3; c++) {
            float val =
                (fracx * fracy * block[c][ceily * (2 * xs + 5) + ceilx] +
                 fracx * (1.f - fracy) *
                     block[c][floory * (2 * xs + 5) + ceilx] +
                 (1.f - fracx) * fracy *
                     block[c][ceily * (2 * xs + 5) + floorx] +
                 (1.f - fracx) * (1.f - fracy) *
                     block[c][floory * (2 * xs + 5) + floorx]) *
                kBlockInvScale * kXybRadius[c];
            if (add) {
              rows[c][iy * stride + ix] += val;
            } else {
              rows[c][iy * stride + ix] -= val;
            }
          }
        }
      }
    }
  }
#ifdef PIK_BD_DUMP_IMAGES
  DumpImage(*opsin);
#endif
}

void BlockDictionary::AddTo(Image3F* opsin, size_t downsampling) const {
  Apply</*add=*/true>(opsin, downsampling);
}

void BlockDictionary::SubtractFrom(Image3F* opsin) const {
  Apply</*add=*/false>(opsin, /*downsampling=*/1);
}

namespace {

float Distance(const QuantizedBlock& a, const Image3F& img, size_t bx,
               size_t by) {
  float dist = 0.0f;
  const size_t stride = img.PixelsPerRow();
  for (size_t c = 0; c < 3; c++) {
    const float* row = img.ConstPlaneRow(c, by) + bx;
    for (size_t iy = 0; iy < a.ysize; iy++) {
      for (size_t ix = 0; ix < a.xsize; ix++) {
        float d = a.pixels[c][iy * a.xsize + ix] -
                  ScaleForQuantization(row[iy * stride + ix], c);
        dist += d * d;
      }
    }
  }
  return dist;
}

constexpr size_t kNumExploreSteps = 1;
constexpr size_t kSmallBlockThreshold = 13;
constexpr float kDistThreshold = 0.6f;
// Returns north, south, east and west neighbors, if present.
size_t Neighbors(size_t x, size_t y, size_t x_max, size_t y_max,
                 std::array<size_t, 2>* neighbors) {
  size_t i = 0;
  if (x != 0) neighbors[i++] = {x - 1, y};
  if (y != 0) neighbors[i++] = {x, y - 1};
  if (x != x_max - 1) neighbors[i++] = {x + 1, y};
  if (y != y_max - 1) neighbors[i++] = {x, y + 1};
  return i;
}

struct Site {
  size_t x;
  size_t y;
  size_t steps;
  constexpr Site(size_t x, size_t y, size_t steps) : x(x), y(y), steps(steps) {}
};

// Finds a bounding box for a loosly connected component. If steps==1, only
// active neighboring pixels are added to the component. If `steps` is larger,
// it is allowed to take a few `steps` through non-active neighboring pixels.
Rect ConnectedCompenentBounds(ImageI* PIK_RESTRICT active, size_t x, size_t y) {
  Rect box = Rect(x, y, 0, 0);
  std::vector<Site> places_to_visit;
  size_t steps = kNumExploreSteps;
  Site site(x, y, steps);
  places_to_visit.emplace_back(site);

  while (!places_to_visit.empty()) {
    Site site = places_to_visit.back();
    x = site.x;
    y = site.y;
    steps = site.steps;
    places_to_visit.pop_back();

    if (steps == 0) continue;
    uint8_t cell_type = active->ConstRow(y)[x];
    if (cell_type == 2) continue;

    if (cell_type == 1) {
      size_t xmin = std::min(x, box.x0());
      size_t ymin = std::min(y, box.y0());
      size_t xmax = std::max(x + 1, box.x0() + box.xsize());
      size_t ymax = std::max(y + 1, box.y0() + box.ysize());
      if (((xmax - xmin) < kMaxBlockSize) && ((ymax - ymin) < kMaxBlockSize)) {
        box = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
      } else {
        continue;
      }
    }
    std::array<size_t, 2> neighbors[4];
    for (int i = 0;
         i < Neighbors(x, y, active->xsize(), active->ysize(), neighbors);
         i++) {
      size_t new_x = neighbors[i][0];
      size_t new_y = neighbors[i][1];
      active->Row(y)[x] = 2;
      site = {new_x, new_y, cell_type ? kNumExploreSteps : steps - 1};
      places_to_visit.emplace_back(site);
    }
  }
  return box;
}
};  // namespace

static const bool kUseBlockDictionary = false;
static const bool kUseHardcodedStretchedBlocks = false;
static const bool KUseDotDetection = true;

BlockDictionary FindBestBlockDictionary(double butteraugli_target,
                                        const Image3F& opsin) {
  if (KUseDotDetection) {
    Image3F without_dots = Image3F(opsin.xsize(), opsin.ysize());
    Image3F dots(opsin.xsize(), opsin.ysize());
    ImageI active(opsin.xsize(), opsin.ysize());
    SplitDots(opsin, &without_dots, &dots);
    std::vector<QuantizedBlock> quantized_blocks;
    std::vector<BlockInfo> blocks;
    std::vector<BlockPosition> positions;
#ifdef PIK_BD_DUMP_IMAGES
    DumpImage(dots);
#endif
    for (size_t y = 0; y < opsin.ysize(); y++) {
      const float* PIK_RESTRICT dot_rows[3];
      const float* PIK_RESTRICT rows[3];
      for (size_t c = 0; c < 3; c++) {
        dot_rows[c] = dots.Plane(c).ConstRow(y);
        rows[c] = without_dots.Plane(c).ConstRow(y);
      }
      int32_t* PIK_RESTRICT active_row = active.Row(y);
      for (size_t x = 0; x < opsin.xsize(); x++) {
        bool is_block = false;
        for (size_t c = 0; c < 3; c++) {
          if (dot_rows[c][x] != 0.0f) {
            is_block = true;
          }
        }
        active_row[x] = is_block;
      }
    }

    for (size_t y = 0; y < opsin.ysize(); y++) {
      const float* PIK_RESTRICT rows[3];
      const float* PIK_RESTRICT dot_rows[3];
      for (size_t c = 0; c < 3; c++) {
        dot_rows[c] = dots.Plane(c).ConstRow(y);
        rows[c] = without_dots.Plane(c).ConstRow(y);
      }
      for (size_t x = 0; x < opsin.xsize(); x++) {
        Rect box = ConnectedCompenentBounds(&active, x, y);
        if (box.xsize() && box.ysize()) {
          BlockInfo fullinfo(&dots, box);
          auto it = std::find_if(blocks.begin(), blocks.end(),
                                 [&](const BlockInfo& other) {
                                   return other.IsSimilar(fullinfo);
                                 });
          if (it == blocks.end()) {
            positions.emplace_back(box.x0(), box.y0(), quantized_blocks.size());
            quantized_blocks.push_back(QuantizedBlock(fullinfo));
            blocks.push_back(fullinfo);
          } else {
            positions.emplace_back(box.x0(), box.y0(), it - blocks.begin());
          }
          ZeroFillImage(&active, box);
        }
      }
    }
    return BlockDictionary{quantized_blocks, positions};
  }
  if (kUseHardcodedStretchedBlocks) {
    std::vector<QuantizedBlock> blocks;
    blocks.push_back(
        QuantizedBlock{1, 3, {{0, 0, 0}, {-11, -8, -11}, {-10, -7, -10}}});
    std::vector<BlockPosition> positions;
    positions.emplace_back(612, 698, 0, 204, 506, -10);
    return BlockDictionary{blocks, positions};
  }
  if (!kUseBlockDictionary) return BlockDictionary{};
  Image3F background_diff(opsin.xsize(), opsin.ysize());
  Image3F background(opsin.xsize(), opsin.ysize());
  const size_t background_stride = background.PixelsPerRow();
  const int kBackgroundBorderPixels = 8;
  std::vector<float> values;
  for (size_t c = 0; c < 3; c++) {
    for (size_t by = 0; by < opsin.ysize(); by++) {
      float* PIK_RESTRICT row_background_diff = background_diff.PlaneRow(c, by);
      float* PIK_RESTRICT row_background = background.PlaneRow(c, by);
      const float* PIK_RESTRICT row_src = opsin.ConstPlaneRow(c, by);
      for (size_t bx = 0; bx < opsin.xsize(); bx++) {
        values.clear();
        for (int iy = -kBackgroundBorderPixels;
             iy < 1 + kBackgroundBorderPixels; iy++) {
          int y = Mirror(by + iy, opsin.ysize());
          const float* PIK_RESTRICT row = opsin.ConstPlaneRow(c, y);
          for (int ix = -kBackgroundBorderPixels;
               ix < 1 + kBackgroundBorderPixels; ix++) {
            int x = Mirror(bx + ix, opsin.xsize());
            values.push_back(row[x]);
          }
        }
        float median = Median(&values);
        row_background[bx] = median;
        row_background_diff[bx] = row_src[bx] - median;
      }
    }
  }

  ImageB used(opsin.xsize(), opsin.ysize());
  FillImage(uint8_t(0), &used);
#ifdef PIK_BD_DUMP_IMAGES
  ImageF shapes(opsin.xsize(), opsin.ysize());
  FillImage(0.0f, &shapes);
#endif

  std::vector<std::array<size_t, 4>> candidates;
  constexpr float kHighPixelThreshold = 0.1f;
  constexpr float kHighPixelSumMultiplier = 5.0f;

  constexpr float kBackgroundThreshold = 0.15f;
  const float* PIK_RESTRICT row_diff = background_diff.ConstPlaneRow(1, 0);
  const size_t stride_diff = background_diff.PixelsPerRow();
  uint8_t* PIK_RESTRICT row_used = used.Row(0);
  const size_t stride_used = used.PixelsPerRow();
#ifdef PIK_BD_DUMP_IMAGES
  float* PIK_RESTRICT row_shapes = shapes.Row(0);
  const size_t stride_shapes = shapes.PixelsPerRow();
  std::mt19937 generator(1);
  std::uniform_real_distribution<double> dis(0.5, 1.0);
#endif
  std::vector<std::pair<int, int>> deltas;
  std::vector<std::pair<int, int>> stack;
  for (size_t by = 0; by < opsin.ysize(); by++) {
    for (size_t bx = 0; bx < opsin.xsize(); bx++) {
      if (row_used[by * stride_used + bx]) continue;
      if (std::fabs(row_diff[by * stride_diff + bx]) < kBackgroundThreshold)
        continue;
      deltas.clear();
      std::pair<int, int> min_delta{};
      std::pair<int, int> max_delta{};
      stack.clear();
      stack.emplace_back(0, 0);
      float sum_high_pixels = 0;
      while (!stack.empty()) {
        std::pair<int, int> delta = stack.back();
        stack.pop_back();
        deltas.push_back(delta);
        if (delta.first < min_delta.first) min_delta.first = delta.first;
        if (delta.first > max_delta.first) max_delta.first = delta.first;
        if (delta.second < min_delta.second) min_delta.second = delta.second;
        if (delta.second > max_delta.second) max_delta.second = delta.second;
        size_t x = bx + delta.first;
        size_t y = by + delta.second;
        if (row_used[y * stride_used + x]) continue;
        row_used[y * stride_used + x] = true;

        if (std::fabs(row_diff[y * stride_diff + x]) > kHighPixelThreshold) {
          sum_high_pixels += std::fabs(row_diff[y * stride_diff + x]);
        }

        int kPixelDeltas[][2] = {{1, 0}, {0, 1},  {-1, 0}, {0, -1},
                                 {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        for (auto dxy : kPixelDeltas) {
          int next_x = x + dxy[0];
          int next_y = y + dxy[1];
          if (next_x < 0 || next_x >= int64_t(opsin.xsize())) continue;
          if (next_y < 0 || next_y >= int64_t(opsin.ysize())) continue;
          if (std::fabs(row_diff[next_y * stride_diff + next_x]) <
              kBackgroundThreshold)
            continue;
          stack.emplace_back(delta.first + dxy[0], delta.second + dxy[1]);
        }
      }
      size_t xsize = max_delta.first - min_delta.first + 1;
      size_t ysize = max_delta.second - min_delta.second + 1;
      if (xsize > kMaxBlockSize) continue;
      if (ysize > kMaxBlockSize) continue;
      if (sum_high_pixels < kHighPixelSumMultiplier * kHighPixelThreshold)
        continue;
      if (min_delta.first + bx + xsize >= opsin.xsize()) continue;
      if (min_delta.second + by + ysize >= opsin.ysize()) continue;
      std::array<size_t, 4> candidate;
      candidate[0] = xsize;
      candidate[1] = ysize;
      candidate[2] = min_delta.first + bx;
      candidate[3] = min_delta.second + by;
      candidates.push_back(candidate);
#ifdef PIK_BD_DUMP_IMAGES
      float val = dis(generator);
      for (std::pair<int, int> delta : deltas) {
        int x = bx + delta.first;
        int y = by + delta.second;
        row_shapes[y * stride_shapes + x] = val;
      }
#endif
    }
  }
  // fprintf(stderr, "%lu\n", candidates.size());
#ifdef PIK_BD_DUMP_IMAGES
  DumpImage(background);
  DumpImage(background_diff);
  Image3F yonly = CopyImage(background_diff);
  FillImage(0.0f, const_cast<ImageF*>(&yonly.Plane(0)));
  FillImage(0.0f, const_cast<ImageF*>(&yonly.Plane(2)));
  DumpImage(yonly);
  DumpImage(shapes);
#endif
  auto extract_block = [&](size_t xsize, size_t ysize, size_t bx, size_t by) {
    QuantizedBlock info;
    info.xsize = xsize;
    info.ysize = ysize;
    for (size_t c = 0; c < 3; c++) {
      for (size_t iy = 0; iy < info.ysize; iy++) {
        const float* row = background_diff.ConstPlaneRow(c, by + iy);
        for (size_t ix = 0; ix < info.xsize; ix++) {
          info.pixels[c][iy * info.xsize + ix] = Quantize(row[bx + ix], c);
        }
      }
    }
    return info;
  };
  auto should_encode = [](const QuantizedBlock& info) {
    size_t num_zeros = 0;
    for (size_t c = 0; c < 3; c++) {
      for (size_t iy = 0; iy < 8; iy++) {
        for (size_t ix = 0; ix < 8; ix++) {
          if (info.pixels[c][iy * 8 + ix] == 0) num_zeros++;
        }
      }
    }
    if (num_zeros == 192) return false;
    return true;
  };
  std::vector<std::pair<size_t, std::array<size_t, 4>>> occurrences;
  // TODO(veluca): take into account off-by-one errors in bounding boxes.
  constexpr size_t kMinNumOccurrences = 4;
  for (size_t i = 0; i < candidates.size(); i++) {
    QuantizedBlock cand = extract_block(candidates[i][0], candidates[i][1],
                                        candidates[i][2], candidates[i][3]);
    if (!should_encode(cand)) continue;
    size_t count = 0;
    for (size_t j = 0; j < candidates.size(); j++) {
      if (i == j) continue;
      if (candidates[j][0] != candidates[i][0]) continue;
      if (candidates[j][1] != candidates[i][1]) continue;
      size_t num_px = candidates[i][0] * candidates[i][1];
      if (Distance(cand, background_diff, candidates[j][2], candidates[j][3]) <
          kDistThreshold * num_px) {
        count++;
      }
    }
    if (count * 2 >= kMinNumOccurrences)
      occurrences.push_back({count, candidates[i]});
  }
  std::sort(occurrences.begin(), occurrences.end(),
            std::greater<std::pair<size_t, std::array<size_t, 4>>>());

  // fprintf(stderr, "%lu\n", occurrences.size());
  // for (size_t i = 0; i < occurrences.size(); i++) {
  //   fprintf(stderr, "%lu ", occurrences[i].first);
  //}
  // fprintf(stderr, "\n");

  std::vector<QuantizedBlock> blocks;
  std::vector<char> taken(occurrences.size());
  for (size_t i = 0; i < occurrences.size(); i++) {
    if (taken[i]) continue;
    QuantizedBlock cand =
        extract_block(occurrences[i].second[0], occurrences[i].second[1],
                      occurrences[i].second[2], occurrences[i].second[3]);
    blocks.push_back(cand);
    for (size_t j = i + 1; j < occurrences.size(); j++) {
      if (occurrences[j].second[0] != occurrences[i].second[0]) continue;
      if (occurrences[j].second[1] != occurrences[i].second[1]) continue;
      size_t num_px = occurrences[i].second[0] * occurrences[i].second[1];
      if (Distance(cand, background_diff, occurrences[j].second[2],
                   occurrences[j].second[3]) < kDistThreshold * num_px) {
        taken[j] = true;
      }
    }
  }

  std::vector<BlockPosition> positions;

  FillImage(uint8_t(0), &used);
  std::vector<size_t> counts(blocks.size());
  std::vector<std::pair<size_t, size_t>> sizes;
  for (size_t i = 0; i < blocks.size(); i++) {
    sizes.emplace_back(blocks[i].xsize, blocks[i].ysize);
  }
  std::sort(sizes.begin(), sizes.end(),
            [](std::pair<size_t, size_t> a, std::pair<size_t, size_t> b) {
              return std::make_pair(a.first * a.second, a.first) >
                     std::make_pair(b.first * b.second, b.first);
            });
  sizes.resize(std::unique(sizes.begin(), sizes.end()) - sizes.begin());
  constexpr float kMaxBackgroundDiff = 0.3f;
  constexpr float kSmallBlockDistThresholdPenalty = 0.3f;
  auto add_matching_block = [&](size_t xs, size_t ys, size_t bx, size_t by) {
    bool overlap = false;
    const float* row_background =
        background.ConstPlaneRow(1, by == 0 ? by : by - 1) +
        (bx == 0 ? bx : bx - 1);
    size_t basex = bx == 0 ? 0 : 1;
    size_t basey = by == 0 ? 0 : 1;
    float background_min_y = row_background[0];
    float background_max_y = row_background[0];
    for (size_t iy = 0; iy < ys; iy++) {
      if (overlap) break;
      const uint8_t* used_row = used.ConstRow(by + iy);
      for (size_t ix = 0; ix < xs; ix++) {
        float cur_y =
            row_background[(basey + iy) * background_stride + basex + ix];
        if (cur_y < background_min_y) background_min_y = cur_y;
        if (cur_y > background_max_y) background_max_y = cur_y;
        if (used_row[bx + ix]) {
          overlap = true;
          break;
        }
      }
    }
    if (overlap) return;
    if (bx != 0 && by != 0 && bx + xs != opsin.xsize() &&
        by + ys != opsin.ysize() && xs * ys < kSmallBlockThreshold) {
      for (size_t iy : {size_t(0), ys + 1}) {
        for (size_t ix = 0; ix < xs + 2; ix++) {
          float cur_y = row_background[iy * background_stride + ix];
          if (cur_y < background_min_y) background_min_y = cur_y;
          if (cur_y > background_max_y) background_max_y = cur_y;
        }
      }
      for (size_t ix : {size_t(0), xs + 1}) {
        for (size_t iy = 0; iy < ys + 2; iy++) {
          float cur_y = row_background[iy * background_stride + ix];
          if (cur_y < background_min_y) background_min_y = cur_y;
          if (cur_y > background_max_y) background_max_y = cur_y;
        }
      }
    }
    float max_background_diff = kMaxBackgroundDiff;
    if (xs * ys < kSmallBlockThreshold)
      max_background_diff *= kSmallBlockDistThresholdPenalty;
    if (background_max_y - background_min_y > max_background_diff) return;
    QuantizedBlock info = extract_block(xs, ys, bx, by);
    if (!should_encode(info)) return;
    size_t id = 0;
    float dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < blocks.size(); i++) {
      if (blocks[i].xsize != xs) continue;
      if (blocks[i].ysize != ys) continue;
      float d = Distance(blocks[i], background_diff, bx, by);
      if (d < dist) {
        id = i;
        dist = d;
      }
    }
    size_t numpx = info.xsize * info.ysize;
    float dist_threshold = kDistThreshold * numpx;
    if (numpx < kSmallBlockThreshold)
      dist_threshold *= kSmallBlockDistThresholdPenalty;
    if (dist > dist_threshold) {
      return;
    }
    if (id == blocks.size()) return;
    for (size_t iy = 0; iy < ys; iy++) {
      uint8_t* used_row = used.Row(by + iy);
      for (size_t ix = 0; ix < xs; ix++) {
        used_row[bx + ix] = true;
      }
    }
    positions.emplace_back(bx, by, id);
    counts[id]++;
  };
  for (auto p : sizes) {
    size_t xs = p.first;
    size_t ys = p.second;
    if (xs * ys < kSmallBlockThreshold) {
      for (size_t i = 0; i < candidates.size(); i++) {
        if (candidates[i][0] != xs || candidates[i][1] != ys) continue;
        add_matching_block(xs, ys, candidates[i][2], candidates[i][3]);
      }
    } else {
      for (size_t by = 0; by < opsin.ysize() - ys; by++) {
        for (size_t bx = 0; bx < opsin.xsize() - xs; bx++) {
          add_matching_block(xs, ys, bx, by);
        }
      }
    }
  }
  std::vector<size_t> remap(blocks.size());
  size_t new_id = 0;
  for (size_t i = 0; i < blocks.size(); i++) {
    remap[i] = new_id;
    if (counts[i] < kMinNumOccurrences) continue;
    blocks[new_id] = blocks[i];
    new_id++;
  }
  blocks.resize(new_id);
  size_t newp = 0;
  for (size_t i = 0; i < positions.size(); i++) {
    if (counts[positions[i].id] < kMinNumOccurrences) continue;
    positions[newp] = positions[i];
    positions[newp].id = remap[positions[newp].id];
    newp++;
  }
  positions.resize(newp);
  new_id = 0;
  for (size_t i = 0; i < counts.size(); i++) {
    if (counts[i] < kMinNumOccurrences) continue;
    counts[new_id++] = counts[i];
  }
  counts.resize(new_id);
  // for (size_t i : counts) fprintf(stderr, "%lu ", i);
  // fprintf(stderr, "\n\n");
  // fprintf(stderr, "%lu %lu\n", blocks.size(), positions.size());
  // for (size_t i = 0; i < blocks.size(); i++) {
  //  fprintf(stderr, "(%lu %lu) ", blocks[i].xsize, blocks[i].ysize);
  //}
  // fprintf(stderr, "\n");
  return BlockDictionary{blocks, positions};
}
}  // namespace pik
