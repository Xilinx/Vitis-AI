// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_UPSCALER_H_
#define PIK_UPSCALER_H_

#include "pik/image.h"

namespace pik {

Image3F Blur(const Image3F& image, float sigma);

}  // namespace pik

#endif  // PIK_UPSCALER_H_
