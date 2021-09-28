// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_GAMMA_CORRECT_H_
#define PIK_GAMMA_CORRECT_H_

// Deprecated: sRGB transfer function. Use color_management.h instead.

#include <cmath>

#include "pik/compiler_specific.h"

namespace pik {

// Values are in [0, 255].
static PIK_INLINE double Srgb8ToLinearDirect(double srgb8) {
  if (srgb8 <= 0.0) return 0.0;
  if (srgb8 <= 10.31475) return srgb8 / 12.92;
  if (srgb8 >= 255.0) return 255.0;
  const double srgb01 = srgb8 / 255.0;
  const double linear01 = std::pow((srgb01 + 0.055) / 1.055, 2.4);
  return linear01 * 255.0;
}

// Values are in [0, 255].
static PIK_INLINE double LinearToSrgb8Direct(double linear) {
  if (linear <= 0.0) return 0.0;
  if (linear >= 255.0) return 255.0;
  if (linear <= 10.31475 / 12.92) return linear * 12.92;
  const double linear01 = linear / 255.0;
  const double srgb01 = std::pow(linear01, 1.0 / 2.4) * 1.055 - 0.055;
  return srgb01 * 255.0;
}

}  // namespace pik

#endif  // PIK_GAMMA_CORRECT_H_
