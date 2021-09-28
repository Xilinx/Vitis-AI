// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_AR_CONTROL_FIELD_H_
#define PIK_AR_CONTROL_FIELD_H_

#include "pik/ac_strategy.h"
#include "pik/image.h"
#include "pik/pik_params.h"
#include "pik/quant_weights.h"

namespace pik {

void FindBestArControlField(float distance, float intensity_target,
                            const Image3F& opsin,
                            const AcStrategyImage& ac_strategy,
                            const ImageF& quant_field,
                            const DequantMatrices* dequant,
                            GaborishStrength gaborish, ThreadPool* pool,
                            ImageB* sigma_lut_ids);

}

#endif  // PIK_AR_CONTROL_FIELD_H_
