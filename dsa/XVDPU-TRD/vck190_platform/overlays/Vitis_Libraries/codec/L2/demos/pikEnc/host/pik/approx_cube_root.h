// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_APPROX_CUBE_ROOT_H_
#define PIK_APPROX_CUBE_ROOT_H_

// Fast cube root for XYB color space.

#include <string.h>

#include "pik/compiler_specific.h"

namespace pik {

PIK_INLINE float CubeRootInitialGuess(float y) {
  int ix;
  memcpy(&ix, &y, sizeof(ix));
  // At this point, ix is the integer value corresponding to the binary
  // representation of the floating point value y. Inspired by the well-known
  // floating-point recipe for 1/sqrt(y), which takes an initial guess in the
  // form of <magic constant> - ix / 2, our initial guess has the form
  // <magic constant> + ix / 3. Since we know the set of all floating
  // point values that will be the input of the cube root function in pik (see
  // LinearToXyb() in opsin_image.cc), we can search for the magic constant that
  // gives the minimum worst-case error. The chosen value here is optimal among
  // the magic constants whose 8 least significant bits are zero.
  ix = 0x2a50f200 + ix / 3;
  float x;
  memcpy(&x, &ix, sizeof(x));
  return x;
}

PIK_INLINE float CubeRootNewtonStep(float y, float xn) {
  constexpr float kOneThird = 1.0f / 3.0f;
  // f(x) = x^3 - y
  // x_{n+1} = x_n - f(x_n) / f'(x_n) =
  //         = x_n - (x_n^3 - y) / (3 * x_n^2) =
  //         = 2/3 * x_n + 1/3 * y / x_n^2
  return kOneThird * (2.0f * xn + y / (xn * xn));
}

// Returns an approximation of the cube root of y,
// with an accuracy of about 1e-6 for 0 <= y <= 1.
PIK_INLINE float ApproxCubeRoot(float y) {
  const float x0 = CubeRootInitialGuess(y);
  const float x1 = CubeRootNewtonStep(y, x0);
  const float x2 = CubeRootNewtonStep(y, x1);
  return x2;
}

}  // namespace pik

#endif  // PIK_APPROX_CUBE_ROOT_H_
