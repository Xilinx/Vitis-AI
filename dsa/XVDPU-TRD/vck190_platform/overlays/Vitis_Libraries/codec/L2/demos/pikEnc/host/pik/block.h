// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_BLOCK_H_
#define PIK_BLOCK_H_

// Adapters for DCT input/output: from/to contiguous blocks or image rows.

#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/simd/simd.h"

namespace pik {

// Adapters for source/destination.
//
// Block: (x, y) <-> (N * y + x)
// Lines: (x, y) <-> (stride * y + x)
//
// I.e. Block is a specialization of Lines with fixed stride.
//
// FromXXX should implement Read and Load (Read vector).
// ToXXX should implement Write and Store (Write vector).

template <size_t N>
using BlockDesc = SIMD_PART(float, SIMD_MIN(N, SIMD_FULL(float)::N));

// Here and in the following, the SZ template parameter specifies the number of
// values to load/store. Needed because we want to handle 4x4 sub-blocks of
// 16x16 blocks.
template <size_t N>
class FromBlock {
 public:
  explicit FromBlock(const float* block) : block_(block) {}

  FromBlock View(size_t dx, size_t dy) const {
    return FromBlock<N>(Address(dx, dy));
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE typename BlockDesc<SZ>::V LoadPart(const size_t row,
                                                          size_t i) const {
    return load(BlockDesc<SZ>(), block_ + row * N + i);
  }

  SIMD_ATTR PIK_INLINE typename BlockDesc<N>::V Load(const size_t row,
                                                     size_t i) const {
    return LoadPart<N>(row, i);
  }

  SIMD_ATTR PIK_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  constexpr PIK_INLINE const float* Address(const size_t row,
                                            const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  const float* block_;
};

template <size_t N>
class ToBlock {
 public:
  explicit ToBlock(float* block) : block_(block) {}

  ToBlock View(size_t dx, size_t dy) const {
    return ToBlock<N>(Address(dx, dy));
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE void StorePart(const typename BlockDesc<SZ>::V& v,
                                      const size_t row, const size_t i) const {
    store(v, BlockDesc<SZ>(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Store(const typename BlockDesc<N>::V& v,
                                  const size_t row, size_t i) const {
    return StorePart<N>(v, row, i);
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr PIK_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  float* block_;
};

// Same as ToBlock, but multiplies result by (N * N)
// TODO(user): perhaps we should get rid of this one.
template <size_t N>
class ScaleToBlock {
 public:
  explicit SIMD_ATTR ScaleToBlock(float* block) : block_(block) {}

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE void StorePart(const typename BlockDesc<SZ>::V& v,
                                      const size_t row, const size_t i) const {
    using BlockDesc = pik::BlockDesc<SZ>;
    static const typename BlockDesc::V mul_ = set1(BlockDesc(), 1.0f / (N * N));
    store(v * mul_, BlockDesc(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Store(const typename BlockDesc<N>::V& v,
                                  const size_t row, size_t i) const {
    return StorePart<N>(v, row, i);
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    static const float mul_ = 1.0f / (N * N);
    *Address(row, i) = v * mul_;
  }

  constexpr PIK_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  float* block_;
};

template <size_t N>
class FromLines {
 public:
  FromLines(const float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  FromLines View(size_t dx, size_t dy) const {
    return FromLines(Address(dx, dy), stride_);
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE typename BlockDesc<SZ>::V LoadPart(
      const size_t row, const size_t i) const {
    return load(BlockDesc<SZ>(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE typename BlockDesc<N>::V Load(const size_t row,
                                                     size_t i) const {
    return LoadPart<N>(row, i);
  }

  SIMD_ATTR PIK_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  PIK_INLINE const float* SIMD_RESTRICT Address(const size_t row,
                                                const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  const float* SIMD_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

// Pointers are restrict-qualified: assumes we don't use both FromLines and
// ToLines in the same DCT. NOTE: Transpose uses From/ToBlock, not *Lines.
template <size_t N>
class ToLines {
 public:
  ToLines(float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  ToLines View(const ToLines& other, size_t dx, size_t dy) const {
    return ToLines(Address(dx, dy), stride_);
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE void StorePart(const typename BlockDesc<SZ>::V& v,
                                      const size_t row, const size_t i) const {
    store(v, BlockDesc<SZ>(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Store(const typename BlockDesc<N>::V& v,
                                  const size_t row, size_t i) const {
    return StorePart<N>(v, row, i);
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    *Address(row, i) = v;
  }

  PIK_INLINE float* SIMD_RESTRICT Address(const size_t row,
                                          const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  float* SIMD_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

}  // namespace pik

#endif  // PIK_BLOCK_H_
