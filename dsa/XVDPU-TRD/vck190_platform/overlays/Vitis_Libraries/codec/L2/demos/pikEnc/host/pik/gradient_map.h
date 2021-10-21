// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_GRADIENT_MAP_H_
#define PIK_GRADIENT_MAP_H_

// The gradient map is a low resolution image (1/8th by 1/8th of the DC, that is
// 1/64th by 1/64th of the image) with finer quantization of the DC. It is used
// to selectively remove banding caused by DC quantization.

#include "pik/compressed_image_fwd.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/padded_bytes.h"
#include "pik/quantizer.h"

namespace pik {

// TODO(robryk): Add unit tests. Verify that
// ComputeGradientMap(ApplyGradientMap(map)) == map.

// For encoding

// Computes the gradient map for the given image of DC
// values.
void ComputeGradientMap(const Image3F& opsin, bool grayscale,
                        const Quantizer& quantizer, GradientMap* gradient);

void SerializeGradientMap(const GradientMap& gradient, const Rect& rect,
                          const Quantizer& quantizer, PaddedBytes* compressed);

// For decoding

Status DeserializeGradientMap(size_t xsize_dc, size_t ysize_dc, bool grayscale,
                              const Quantizer& quantizer,
                              const PaddedBytes& compressed, size_t* byte_pos,
                              GradientMap* gradient);

// Applies the gradient map to the decoded DC image.
void ApplyGradientMap(const GradientMap& gradient, const Quantizer& quantizer,
                      Image3F* opsin);

}  // namespace pik

#endif  // PIK_GRADIENT_MAP_H_
