// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ADAPTIVE_RECONSTRUCTION_FWD_H_
#define PIK_ADAPTIVE_RECONSTRUCTION_FWD_H_

// Breaks the circular dependency between adaptive_reconstruction.h and
// pik_info.h.

#include "pik/epf_stats.h"
#include "pik/image.h"

namespace pik {

// Optional output(s).
struct AdaptiveReconstructionAux {
  void Assimilate(const AdaptiveReconstructionAux& other) {
    epf_stats.Assimilate(other.epf_stats);
    if (other.stretch != -1.0f) stretch = other.stretch;
    if (other.quant_scale != -1.0f) quant_scale = other.quant_scale;
  }

  void Print() const { epf_stats.Print(); }

  // Filled with the multiplier used to scale input pixels to [0, 255].
  float stretch = -1.0f;

  // Set to Quantizer::Scale().
  float quant_scale = -1.0f;

  // If not null, filled with difference between input and filtered image.
  Image3F* residual = nullptr;
  // If not null, filled with the output of the filter.
  Image3F* filtered = nullptr;
  // If not null, filled with raw quant map used to compute sigma.
  ImageI* ac_quant = nullptr;
  // If not null, filled with AC strategy (for detecting DCT16)
  ImageB* ac_strategy = nullptr;

  EpfStats epf_stats;
};

}  // namespace pik

#endif  // PIK_ADAPTIVE_RECONSTRUCTION_FWD_H_
