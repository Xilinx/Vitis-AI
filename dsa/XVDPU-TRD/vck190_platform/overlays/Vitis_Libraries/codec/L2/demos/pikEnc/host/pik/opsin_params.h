// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_OPSIN_PARAMS_H_
#define PIK_OPSIN_PARAMS_H_

// Constants that define the XYB color space.

#include <stdlib.h>

#include "pik/simd/simd.h"  // SIMD_ALIGN

namespace pik {

static constexpr float kScale = 255.0f;

// NOTE: inverse of this cannot be constant because we tune these values.
static const float kOpsinAbsorbanceMatrix[9] = {
    static_cast<float>((0.29956550340058319) / kScale),
    static_cast<float>((0.63373087833825936) / kScale),
    static_cast<float>((0.077705617820981968) / kScale),
    static_cast<float>((0.22158691104574774) / kScale),
    static_cast<float>((0.68491388044116142) / kScale),
    static_cast<float>((0.096254234043612538) / kScale),
    static_cast<float>((0.20062661225219422) / kScale),
    static_cast<float>((0.070366199217588729) / kScale),
    static_cast<float>((0.5571760754215358) / kScale),
};

// Returns 3x3 row-major matrix inverse of kOpsinAbsorbanceMatrix.
// opsin_image_test verifies this is actually the inverse.
const float* GetOpsinAbsorbanceInverseMatrix();

static const float kOpsinAbsorbanceBias[3] = {
    static_cast<float>((0.26786006338144885) / kScale),
    static_cast<float>((0.24494032763907073) / kScale),
    static_cast<float>((0.14255999980363571) / kScale),
};
SIMD_ALIGN static const float kNegOpsinAbsorbanceBiasRGB[4] = {
    -kOpsinAbsorbanceBias[0], -kOpsinAbsorbanceBias[1],
    -kOpsinAbsorbanceBias[2], 255.0f};

static const float kScaleR = 1.0f;
static const float kScaleG = 2.0f - kScaleR;
static const float kInvScaleR = 1.0f / kScaleR;
static const float kInvScaleG = 1.0f / kScaleG;

static constexpr float kXybCenter[3] = {0.0087982f, 0.5513899f, 0.4716444f};

// Radius of the XYB range around the center. The full range is 2 * kXybRadius.
static constexpr float kXybRadius[3] = {0.0301006f, 0.4512295f, 0.4716444f};

static constexpr float kXybMin[3] = {
    kXybCenter[0] - kXybRadius[0],
    kXybCenter[1] - kXybRadius[1],
    kXybCenter[2] - kXybRadius[2],
};
static constexpr float kXybMax[3] = {
    kXybCenter[0] + kXybRadius[0],
    kXybCenter[1] + kXybRadius[1],
    kXybCenter[2] + kXybRadius[2],
};

}  // namespace pik

#endif  // PIK_OPSIN_PARAMS_H_
