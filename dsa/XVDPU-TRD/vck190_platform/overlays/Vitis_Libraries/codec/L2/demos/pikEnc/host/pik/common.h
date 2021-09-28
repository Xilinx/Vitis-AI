// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COMMON_H_
#define PIK_COMMON_H_

// Shared constants and helper functions.

#include <stddef.h>
#include <memory>  // unique_ptr

#include "pik/compiler_specific.h"

namespace pik {

constexpr size_t kBitsPerByte = 8;  // more clear than CHAR_BIT

template <typename T>
constexpr inline T DivCeil(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
constexpr T Pi(T multiplier) {
  return static_cast<T>(multiplier * 3.1415926535897932);
}

// Block is the square grid of pixels to which an "energy compaction"
// transformation (e.g. DCT) is applied. Each block has its own AC quantizer.
constexpr size_t kBlockDim = 8;

constexpr size_t kDCTBlockSize = kBlockDim * kBlockDim;

// Group is the rectangular grid of blocks that can be decoded in parallel. This
// is different for DC.
constexpr size_t kDcGroupDimInBlocks = 256;
constexpr size_t kGroupDim = 512;
static_assert(kGroupDim % kBlockDim == 0,
              "Group dim should be divisible by block dim");
constexpr size_t kGroupDimInBlocks = kGroupDim / kBlockDim;

// We split groups into tiles to increase locality and cache hits.
const constexpr size_t kTileDim = 64;

static_assert(kTileDim % kBlockDim == 0,
              "Tile dim should be divisible by block dim");
constexpr size_t kTileDimInBlocks = kTileDim / kBlockDim;

static_assert(kGroupDimInBlocks % kTileDimInBlocks == 0,
              "Group dim should be divisible by tile dim");

// Can't rely on C++14 yet.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// This leads to somewhat better code than pointer arithmetic.
template <typename T>
PIK_INLINE T* PIK_RESTRICT ByteOffset(T* PIK_RESTRICT base,
                                      const intptr_t byte_offset) {
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
  return reinterpret_cast<T*>(base_addr + byte_offset);
}

}  // namespace pik

#endif  // PIK_COMMON_H_
