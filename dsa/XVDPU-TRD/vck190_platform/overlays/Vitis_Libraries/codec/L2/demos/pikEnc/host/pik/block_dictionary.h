// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_BLOCK_DICTIONARY_H_
#define PIK_BLOCK_DICTIONARY_H_

// Chooses reference blocks out of the image, and sets things up to avoid
// encoding once for each occurrence.

#include <cstddef>
#include "pik/bit_reader.h"
#include "pik/image.h"
#include "pik/opsin_params.h"
#include "pik/pik_info.h"

#include <vector>

namespace pik {

constexpr size_t kMaxBlockSize = 8;
constexpr float kDotDistThreshold = 0.02f;

struct QuantizedBlock {
  size_t xsize;
  size_t ysize;
  int8_t pixels[3][kMaxBlockSize * kMaxBlockSize];
  bool operator==(const QuantizedBlock& other) const {
    if (xsize != other.xsize) return false;
    if (ysize != other.ysize) return false;
    for (size_t c = 0; c < 3; c++) {
      if (memcmp(pixels[c], other.pixels[c], sizeof(int8_t) * xsize * ysize) !=
          0)
        return false;
    }
    return true;
  }
  bool operator<(const QuantizedBlock& other) const {
    if (xsize < other.xsize) return true;
    if (xsize > other.xsize) return false;
    if (ysize < other.ysize) return true;
    if (ysize > other.ysize) return false;
    for (size_t c = 0; c < 3; c++) {
      int cmp =
          memcmp(pixels[c], other.pixels[c], sizeof(int8_t) * xsize * ysize);
      if (cmp > 0) return false;
      if (cmp < 0) return true;
    }
    return false;
  }
};
struct BlockPosition {
  // Position of top-left corner of the block in the image.
  size_t x, y;
  size_t id;
  bool transform = false;
  // Offset of top-right corner from top-left one.
  int64_t dx = 0;
  int64_t dy = 0;
  // Measured in half-pixels.
  int64_t width = 0;
  BlockPosition() {}
  BlockPosition(size_t x, size_t y, size_t id) : x(x), y(y), id(id) {}
  BlockPosition(size_t x, size_t y, size_t id, int64_t dx, int64_t dy,
                int64_t width)
      : x(x), y(y), id(id), transform(true), dx(dx), dy(dy) {}
};

class BlockDictionary {
 public:
  BlockDictionary() {}
  BlockDictionary(const std::vector<QuantizedBlock>& dictionary,
                  const std::vector<BlockPosition>& positions);

  std::string Encode(PikImageSizeInfo* info) const;

  Status Decode(BitReader* br, size_t xsize, size_t ysize);

  void AddTo(Image3F* opsin, size_t downsampling) const;

  void SubtractFrom(Image3F* opsin) const;

 private:
  std::vector<QuantizedBlock> dictionary_;
  std::vector<BlockPosition> positions_;
  template <bool>
  void Apply(Image3F* opsin, size_t downsampling) const;
};

BlockDictionary FindBestBlockDictionary(double butteraugli_target,
                                        const Image3F& opsin);

}  // namespace pik

#endif  // PIK_BLOCK_DICTIONARY_H_
