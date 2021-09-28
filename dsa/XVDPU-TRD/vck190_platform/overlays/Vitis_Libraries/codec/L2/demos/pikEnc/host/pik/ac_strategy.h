// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_AC_STRATEGY_H_
#define PIK_AC_STRATEGY_H_

#include <stdint.h>
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/dct.h"
#include "pik/dct_util.h"
#include "pik/image.h"
#include "pik/pik_info.h"
#include "pik/quant_weights.h"

// Defines the different kinds of transforms, and heuristics to choose between
// them.
// `AcStrategy` represents what transform should be used, and which sub-block of
// that transform we are currently in. Note that DCT4x4 is applied on all four
// 4x4 sub-blocks of an 8x8 block.
// `AcStrategyImage` defines which strategy should be used for each 8x8 block
// of the image. The highest 4 bits represent the strategy to be used, the
// lowest 4 represent the index of the block inside that strategy. Blocks should
// be aligned, i.e. 32x32 blocks should only start in positions that are
// multiples of 32.
// `FindBestAcStrategy` uses heuristics to choose which AC strategy should be
// used in each block.

namespace pik {
namespace detail {
template <size_t SZ, size_t BX, size_t BY>
static constexpr float ARLowestFrequencyScale() {
  return SZ * IDCTScales<SZ>()[BX] * IDCTScales<SZ>()[BY] * L1Norm<SZ>()[BX] *
         L1Norm<SZ>()[BY];
}
}  // namespace detail

class AcStrategy {
 public:
  // Extremal values for the number of blocks/coefficients of a single strategy.
  static constexpr size_t kMaxCoeffBlocks = 4;
  static constexpr size_t kMaxBlockDim = kBlockDim * kMaxCoeffBlocks;
  static constexpr size_t kMaxCoeffArea = kMaxBlockDim * kMaxBlockDim;
  static constexpr size_t kLLFMaskDim =
      DivCeil(kMaxBlockDim, SIMD_FULL(float)::N) * SIMD_FULL(float)::N;

  // Raw strategy types.
  enum class Type : uint32_t {
    // Regular block size DCT (value matches kQuantKind)
    DCT = 0,
    // Encode pixels without transforming (value matches kQuantKind)
    IDENTITY = 1,
    // Use 2-by-2 DCT (value matches kQuantKind)
    DCT2X2 = 2,
    // Use 4-by-4 DCT (value matches kQuantKind)
    DCT4X4 = 3,
    // Use 16-by-16 DCT
    DCT16X16 = 4,
    // Use 32-by-32 DCT
    DCT32X32 = 5,
    // Angled lines (currently hardcoded for 45 degrees)
    LINES = 6,
    // Use 8-by-8 DCT, no HF prediction
    DCT_NOHF = 7,
    // Use 4-by-4 DCT, no HF prediction
    DCT4X4_NOHF = 8,
  };

  PIK_INLINE AcStrategy(Type strategy, uint32_t block)
      : strategy_(strategy), block_(block) {
#ifdef ADDRESS_SANITIZER
    PIK_ASSERT(strategy == Type::DCT16X16 || strategy == Type::DCT32X32 ||
               block == 0);
    PIK_ASSERT(strategy == Type::DCT32X32 || block < 4);
    PIK_ASSERT(block < 16);
#endif
  }

  // Returns true if this block is the first 8x8 block (i.e. top-left) of a
  // possibly multi-block strategy.
  PIK_INLINE bool IsFirstBlock() const { return block_ == 0; }

  // Returns the raw strategy value. Should only be used for tokenization.
  PIK_INLINE uint8_t RawStrategy() const {
    return static_cast<uint8_t>(strategy_);
  }

  PIK_INLINE Type Strategy() const { return strategy_; }
  PIK_INLINE size_t Block() const { return block_; }

  // Inverse check
  static PIK_INLINE bool IsRawStrategyValid(uint8_t raw_strategy) {
    return raw_strategy <= 8;
  }
  static PIK_INLINE AcStrategy FromRawStrategy(uint8_t raw_strategy) {
    return AcStrategy((Type)raw_strategy, 0);
  }

  // Get the quant kind for this type of strategy.
  PIK_INLINE size_t GetQuantKind(size_t block = 0) const {
    static_assert(kMaxCoeffArea == kMaxQuantTableSize,
                  "Maximum coefficient area should be the same as maximum "
                  "quant table size!");
    if (strategy_ == Type::DCT_NOHF) return kQuantKindDCT8;
    if (strategy_ == Type::DCT4X4_NOHF) return kQuantKindDCT4;

    static_assert(kQuantKindDCT8 == size_t(Type::DCT), "QuantKind != type");
    static_assert(kQuantKindID == size_t(Type::IDENTITY), "QuantKind != type");
    static_assert(kQuantKindDCT4 == size_t(Type::DCT4X4), "QuantKind != type");
    static_assert(kQuantKindDCT2 == size_t(Type::DCT2X2), "QuantKind != type");
    static_assert(kQuantKindDCT16 == size_t(Type::DCT16X16),
                  "QuantKind != type");
    static_assert(kQuantKindDCT32 == size_t(Type::DCT32X32),
                  "QuantKind != type");
    static_assert(kQuantKindLines == size_t(Type::LINES), "QuantKind != type");
    return static_cast<size_t>(strategy_);
  }

  PIK_INLINE float ARQuantScale() const {
    if (strategy_ == Type::DCT32X32) return 1.2282996852328099;
    if (strategy_ == Type::DCT16X16) return 1.1423171621463439;
    // TODO(veluca): find better values.
    if (strategy_ == Type::DCT4X4 || strategy_ == Type::DCT4X4_NOHF)
      return 1.2f;
    if (strategy_ == Type::DCT2X2) return 1.0098134203870499;
    return 1.0f;
  }

  PIK_INLINE bool PredictHF() const {
    return strategy_ != Type::DCT2X2 && strategy_ != Type::IDENTITY &&
           strategy_ != Type::DCT_NOHF && strategy_ != Type::DCT4X4_NOHF &&
           strategy_ != Type::LINES;
  }

  // Number of 8x8 blocks that this strategy will cover. 0 for non-top-left
  // blocks inside a multi-block transform.
  PIK_INLINE size_t covered_blocks_x() const {
    if (strategy_ == Type::DCT32X32) return block_ == 0 ? 4 : 0;
    if (strategy_ == Type::DCT16X16) return block_ == 0 ? 2 : 0;
    return 1;
  }
  PIK_INLINE size_t covered_blocks_y() const {
    if (strategy_ == Type::DCT32X32) return block_ == 0 ? 4 : 0;
    if (strategy_ == Type::DCT16X16) return block_ == 0 ? 2 : 0;
    return 1;
  }

  // 1 / covered_block_x() / covered_block_y(), for fast division.
  // Should only be called with block_ == 0.
  PIK_INLINE float inverse_covered_blocks() const {
#ifdef ADDRESS_SANITIZER
    PIK_ASSERT(block_ == 0);
#endif
    if (strategy_ == Type::DCT32X32) return 1.0f / 16;
    if (strategy_ == Type::DCT16X16) return 0.25f;
    return 1.0f;
  }

  PIK_INLINE float InverseNumACCoefficients() const {
#ifdef ADDRESS_SANITIZER
    PIK_ASSERT(block_ == 0);
#endif
    if (strategy_ == Type::DCT32X32) return 1.0f / (32 * 32 - 16);
    if (strategy_ == Type::DCT16X16) return 1.0f / (16 * 16 - 4);
    return 1.0f / (8 * 8 - 1);
  }

  const float* ARLowestFrequencyScales(size_t y) {
    using detail::ARLowestFrequencyScale;
    switch (strategy_) {
      case Type::DCT2X2:
      case Type::IDENTITY:
      case Type::DCT:
      case Type::DCT4X4:
      case Type::DCT_NOHF:
      case Type::DCT4X4_NOHF:
      case Type::LINES: {
        SIMD_ALIGN static const constexpr float scales[kLLFMaskDim] = {1.0f};
        return scales;
      }
      case Type::DCT16X16: {
        SIMD_ALIGN static const constexpr float scales[2][kLLFMaskDim] = {
            {ARLowestFrequencyScale<2 * kBlockDim, 0, 0>(),
             ARLowestFrequencyScale<2 * kBlockDim, 1, 0>()},
            {ARLowestFrequencyScale<2 * kBlockDim, 0, 1>(),
             ARLowestFrequencyScale<2 * kBlockDim, 1, 1>()}};
        return scales[y];
      }
      case Type::DCT32X32: {
        SIMD_ALIGN static const constexpr float scales[4][kLLFMaskDim] = {
            {
                ARLowestFrequencyScale<4 * kBlockDim, 0, 0>(),
                ARLowestFrequencyScale<4 * kBlockDim, 1, 0>(),
                ARLowestFrequencyScale<4 * kBlockDim, 2, 0>(),
                ARLowestFrequencyScale<4 * kBlockDim, 3, 0>(),
            },
            {
                ARLowestFrequencyScale<4 * kBlockDim, 0, 1>(),
                ARLowestFrequencyScale<4 * kBlockDim, 1, 1>(),
                ARLowestFrequencyScale<4 * kBlockDim, 2, 1>(),
                ARLowestFrequencyScale<4 * kBlockDim, 3, 1>(),
            },
            {
                ARLowestFrequencyScale<4 * kBlockDim, 0, 2>(),
                ARLowestFrequencyScale<4 * kBlockDim, 1, 2>(),
                ARLowestFrequencyScale<4 * kBlockDim, 2, 2>(),
                ARLowestFrequencyScale<4 * kBlockDim, 3, 2>(),
            },
            {
                ARLowestFrequencyScale<4 * kBlockDim, 0, 3>(),
                ARLowestFrequencyScale<4 * kBlockDim, 1, 3>(),
                ARLowestFrequencyScale<4 * kBlockDim, 2, 3>(),
                ARLowestFrequencyScale<4 * kBlockDim, 3, 3>(),
            }};
        return scales[y];
      }
    }
    SIMD_ALIGN static const constexpr float scales[kLLFMaskDim] = {1.0f};
    return scales;
  }

  // Pixel to coefficients and vice-versa
  SIMD_ATTR void TransformFromPixels(const float* PIK_RESTRICT pixels,
                                     size_t pixels_stride,
                                     float* PIK_RESTRICT coefficients,
                                     size_t coefficients_stride) const;
  SIMD_ATTR void TransformToPixels(const float* PIK_RESTRICT coefficients,
                                   size_t coefficients_stride,
                                   float* PIK_RESTRICT pixels,
                                   size_t pixels_stride) const;

  // Coefficient scattering and gathering.
  template <typename T>
  SIMD_ATTR void ScatterCoefficients(const T* PIK_RESTRICT coefficients,
                                     size_t coefficients_stride,
                                     T* PIK_RESTRICT blocks,
                                     size_t blocks_stride) const {
    if (block_ != 0) return;
    if (covered_blocks_x() == 4 && covered_blocks_y() == 4) {
      ScatterBlock<4 * kBlockDim, 4 * kBlockDim>(
          coefficients, coefficients_stride, blocks, blocks_stride);
      return;
    }
    if (covered_blocks_x() == 2 && covered_blocks_y() == 2) {
      ScatterBlock<2 * kBlockDim, 2 * kBlockDim>(
          coefficients, coefficients_stride, blocks, blocks_stride);
      return;
    }
    PIK_ASSERT(covered_blocks_x() == 1 && covered_blocks_y() == 1);
    memcpy(blocks, coefficients, kBlockDim * kBlockDim * sizeof(T));
  }

  template <typename T>
  SIMD_ATTR void GatherCoefficients(const T* PIK_RESTRICT blocks,
                                    size_t blocks_stride,
                                    T* PIK_RESTRICT coefficients,
                                    size_t coefficients_stride) const {
    if (block_ != 0) return;
    if (covered_blocks_x() == 4 && covered_blocks_y() == 4) {
      GatherBlock<4 * kBlockDim, 4 * kBlockDim>(
          blocks, blocks_stride, coefficients, coefficients_stride);
      return;
    }
    if (covered_blocks_x() == 2 && covered_blocks_y() == 2) {
      GatherBlock<2 * kBlockDim, 2 * kBlockDim>(
          blocks, blocks_stride, coefficients, coefficients_stride);
      return;
    }
    PIK_ASSERT(covered_blocks_x() == 1 && covered_blocks_y() == 1);
    memcpy(coefficients, blocks, kBlockDim * kBlockDim * sizeof(T));
  }

  // Same as above, but for DC image.
  SIMD_ATTR void LowestFrequenciesFromDC(const float* PIK_RESTRICT dc,
                                         size_t dc_stride, float* llf,
                                         size_t llf_stride) const;
  SIMD_ATTR void DCFromLowestFrequencies(const float* PIK_RESTRICT block,
                                         size_t block_stride, float* dc,
                                         size_t dc_stride) const;

  // Produces a 2x2-upsampled DC block out of the lowest frequencies
  // (block_size/8) of the image.
  SIMD_ATTR void DC2x2FromLowestFrequencies(const float* PIK_RESTRICT llf,
                                            size_t llf_stride,
                                            float* PIK_RESTRICT dc2x2,
                                            size_t dc2x2_stride) const;

  // Produces the low frequencies (block_size/4) of the images out of a 2x2
  // upsampled DC image, and vice-versa.
  SIMD_ATTR void DC2x2FromLowFrequencies(const float* block,
                                         size_t block_stride, float* dc2x2,
                                         size_t dc2x2_stride) const;
  SIMD_ATTR void LowFrequenciesFromDC2x2(const float* dc2x2,
                                         size_t dc2x2_stride, float* block,
                                         size_t block_stride) const;

 private:
  Type strategy_;
  uint32_t block_;
};  // namespace pik

// Class to use a certain row of the AC strategy.
class AcStrategyRow {
 public:
  AcStrategyRow(const uint8_t* row, size_t y) : row_(row) {}
  AcStrategy operator[](size_t x) const {
    return AcStrategy((AcStrategy::Type)(row_[x] >> 4), row_[x] & 0xF);
  }

 private:
  const uint8_t* PIK_RESTRICT row_;
};

class AcStrategyImage {
 public:
  // A value that does not represent a valid combined AC strategy value.
  // Used as a sentinel in DecodeAcStrategy.
  static constexpr uint8_t INVALID = 0xF;

  AcStrategyImage() {}
  AcStrategyImage(size_t xsize, size_t ysize) : layers_(xsize, ysize) {
    FillImage((uint8_t)AcStrategy::Type::DCT, &layers_);
  }
  AcStrategyImage(AcStrategyImage&&) = default;
  AcStrategyImage& operator=(AcStrategyImage&&) = default;

  // `rect` is the area to fill with the entire contents of `raw_layers`.
  void SetFromRaw(const Rect& rect, const ImageB& raw_layers);

  void SetFromArray(const Rect& rect, uint32_t data[]);

  AcStrategyRow ConstRow(size_t y, size_t x_prefix = 0) const {
    return AcStrategyRow(layers_.ConstRow(y) + x_prefix, y);
  }

  AcStrategyRow ConstRow(const Rect& rect, size_t y) const {
    return ConstRow(rect.y0() + y, rect.x0());
  }

  const ImageB& ConstRaw() const { return layers_; }

  size_t xsize() const { return layers_.xsize(); }
  size_t ysize() const { return layers_.ysize(); }

  AcStrategyImage Copy(const Rect& rect) const {
    AcStrategyImage copy;
    copy.layers_ = CopyImage(rect, layers_);
    return copy;
  }

  AcStrategyImage Copy() const { return Copy(Rect(layers_)); }

  // Count the number of blocks of a given type.
  size_t CountBlocks(AcStrategy::Type type) const;

 private:
  ImageB layers_;
};

// `quant_field` is an initial quantization field for this image. `src` is the
// input image in the XYB color space. `ac_strategy` is the output strategy.
SIMD_ATTR void FindBestAcStrategy(float butteraugli_target,
                                  const ImageF* quant_field,
                                  const DequantMatrices& dequant,
                                  const Image3F& src, ThreadPool* pool,
                                  AcStrategyImage* ac_strategy,
                                  PikInfo* aux_out);

}  // namespace pik

#endif  // PIK_AC_STRATEGY_H_
