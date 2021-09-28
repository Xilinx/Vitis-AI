// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_EPF_H_
#define PIK_EPF_H_

// Fast SIMD edge preserving filter (adaptive, nonlinear).

#include <stdio.h>
#include "pik/ac_strategy.h"
#include "pik/field_encodings.h"
#include "pik/image.h"

namespace pik {

struct EpfParams {
  EpfParams();
  static const char* Name() { return "EpfParams"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    visitor->Bool(false/*true*/, &enable_adaptive);
    if (visitor->Conditional(!enable_adaptive)) {
      visitor->U32(0x0A090880, 0, &sigma);
    }
    visitor->Bool(false, &use_sharpened);
    return true;
  }

  // If false, use hardcoded sigma for each block.
  bool enable_adaptive;

  // Only if !enable_adaptive:
  uint32_t sigma;  // ignored if !enable_adaptive, otherwise >= kMinSigma.

  bool use_sharpened;
};

// Unit test. Call via dispatch::ForeachTarget.
struct EdgePreservingFilterTest {
  template <class Target>
  void operator()() const;

  // Returns weight given sigma and SAD.
  template <class Target>
  float operator()(int sigma, int sad) const;
};

// Must be called before EdgePreservingFilter, with the same Target.
struct InitEdgePreservingFilter {
  template <class Target>
  void operator()() const;
};

// Adaptive smoothing based on quantization intervals. "sigma" must be in
// [kMinSigma, kMaxSigma]. Fills each pixel of "smoothed", which must be
// pre-allocated. Call via Dispatch.
struct EdgePreservingFilter {
  // The "sigma" parameter is the SCALED half-width at half-maximum, i.e. the
  // SAD value for which the weight is 0.5, times the scaling factor of
  // 1 << kSigmaShift. Before scaling, sigma is about 1.2 times the standard
  // deviation of a normal distribution. Larger values cause more smoothing.

  // All sigma values are pre-shifted by this value to increase their
  // resolution. This allows adaptive sigma to compute "5.5" (represented as 22)
  // without an additional floating-point multiplication.
  static constexpr int kSigmaShift = 2;

  // This is the smallest value that avoids 16-bit overflow (see kShiftSAD); it
  // corresponds to 1/3 of patch pixels having the minimum integer SAD of 1.
  static constexpr int kMinSigma = 4 << kSigmaShift;
  // Somewhat arbitrary; determines size of a lookup table.
  static constexpr int kMaxSigma = 168 << kSigmaShift;  // 14 per patch pixel

  // For each block, compute adaptive sigma.
  template <class Target>
  void operator()(const Image3F& in_guide, const Image3F& in,
                  const ImageI* ac_quant, float quant_scale,
                  const ImageB& lut_ids, const AcStrategyImage& ac_strategy,
                  const EpfParams& epf_params,
                  Image3F* smoothed, EpfStats* epf_stats) const;

  // Fixed sigma in [kMinSigma, kMaxSigma] for generating training data;
  // sigma == 0 skips filtering and copies "in" to "smoothed".
  // "stretch" is returned for use by AdaptiveReconstructionAux.
  template <class Target>
  void operator()(const Image3F& in_guide, const Image3F& in,
                  const EpfParams& params, float* PIK_RESTRICT stretch,
                  Image3F* smoothed) const;
};

}  // namespace pik

#endif  // PIK_EPF_H_
