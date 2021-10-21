// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// We attempt to remove dots, or speckle from images using Gaussian blur.
#ifndef PIK_RESEARCH_REMOVE_DOTS_H_
#define PIK_RESEARCH_REMOVE_DOTS_H_

#include <cstdio>
#include <string>

#include "pik/codec.h"
#include "pik/data_parallel.h"
#include "pik/file_io.h"
#include "pik/gauss_blur.h"
#include "pik/image.h"
#include "pik/opsin_image.h"
#include "pik/opsin_inverse.h"
#include "pik/opsin_params.h"

namespace pik {

// Detects dots in an given `image` and splits the `image` in two images:
// - `dots`: containing only the dots and
// - `without_dots`, containing the original image, where the dots have been
// replaced with the median of surrounding pixels.
void SplitDots(const Image3F& image, Image3F* without_dots, Image3F* dots);

}  // namespace pik

#endif  // PIK_RESEARCH_REMOVE_DOTS_H_
