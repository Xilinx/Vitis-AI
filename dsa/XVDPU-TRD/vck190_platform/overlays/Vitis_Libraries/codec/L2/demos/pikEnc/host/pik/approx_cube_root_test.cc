// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/approx_cube_root.h"

#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"
#include "pik/gamma_correct.h"
#include "pik/opsin_image.h"

namespace {

TEST(ApproxCubeRootTest, MaxError) {
  float maxerr = 0.0f;
  // (Runtime about 16 seconds)
  for (float r = 0; r <= 255.0f; r += 0.5f) {
    for (float g = 0; g <= 255.0f; g += 0.5f) {
      for (float b = 0; b <= 255.0f; b += 0.5f) {
        float mixed[3];
        pik::OpsinAbsorbance(r, g, b, mixed);
        for (int c = 0; c < 3; ++c) {
          float x = mixed[c];
          float error = std::abs(pik::ApproxCubeRoot(x) - std::cbrt(x));
          maxerr = std::max(maxerr, error);
        }
      }
    }
  }
  EXPECT_LT(maxerr, 2E-6f);
}

#if FIND_BEST_MAGIC_CONSTANT

// Parameterized versions CubeRootInitialGuess() and ApproxCubeRoot(),
// where the parameter is the magic constant used to find the initial guess.
PIK_INLINE float CubeRootInitialGuessP(uint32_t magic, float y) {
  int ix;
  memcpy(&ix, &y, sizeof(ix));
  ix = magic + ix / 3;
  float x;
  memcpy(&x, &ix, sizeof(x));
  return x;
}

PIK_INLINE float ApproxCubeRootP(uint32_t magic, float y) {
  const float x0 = CubeRootInitialGuessP(magic, y);
  const float x1 = pik::CubeRootNewtonStep(y, x0);
  const float x2 = pik::CubeRootNewtonStep(y, x1);
  return x2;
}

// Compute the maximum error over the opsin dynamics color space for the given
// magic constant. If the error for an (r,g,b) value is over cutoff, we stop
// early and return cutoff. This is because we only need the magic constant
// with the minimum worst-case error.
float MaxErrorP(uint32_t magic, float cutoff) {
  float maxerr = 0.0f;
  for (float r = 0; r <= 255.0f; r += 0.5f) {
    for (float g = 0; g <= 255.0f; g += 0.5f) {
      for (float b = 0; b <= 255.0f; b += 0.5f) {
        const float rgb[3] = {r, g, b};
        float mixed[3];
        pik::OpsinAbsorbance(rgb, mixed);
        for (int c = 0; c < 3; ++c) {
          float x = mixed[c];
          float error = std::abs(ApproxCubeRootP(magic, x) - std::cbrt(x));
          maxerr = std::max(maxerr, error);
        }
        if (maxerr > cutoff) return cutoff;
      }
    }
  }
  return maxerr;
}

TEST(ApproxCubeRootTest, FindBestMagicConstant) {
  float min_max_err = 1e-6f;
  uint32_t best_magic = 0;
  // We only search for the 24 most significant bits of the magic constant,
  // since the lower 8 bits only influence the initial guess by 2^(8-23) ~ 3e-4.
  for (uint64_t magic = 0; magic <= 0xffffffffULL; magic += 0x100) {
    float max_err = MaxErrorP(magic, min_max_err);
    if (max_err < min_max_err) {
      min_max_err = max_err;
      best_magic = magic;
    }
  }
  printf("Best magic constant: 0x%08x,  max approx error: %.012f\n", best_magic,
         min_max_err);
}

#endif  // FIND_BEST_MAGIC_CONSTANT

}  // namespace
