// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/image.h"

namespace pik {

constexpr size_t kNumBilinearParams = 8;

struct BilinearParams {
  BilinearParams(int xtiles, int ytiles)
      : transform_params(xtiles * ytiles * kNumBilinearParams),
        is_transform_applied(xtiles * ytiles) {}

  std::vector<double> transform_params;
  std::vector<bool> is_transform_applied;
};

BilinearParams ApplyReverseBilinear(Image3F *opsin);
void ApplyForwardBilinear(Image3F *opsin, size_t downsample);

}  // namespace pik
