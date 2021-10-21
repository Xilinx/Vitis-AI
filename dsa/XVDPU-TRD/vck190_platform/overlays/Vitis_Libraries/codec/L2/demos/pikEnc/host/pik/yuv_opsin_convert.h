// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_YUV_OPSIN_CONVERT_H_
#define PIK_YUV_OPSIN_CONVERT_H_

#include "pik/image.h"

namespace pik {

Image3B RGB8ImageFromYUVOpsin(const Image3U& yuv, int bit_depth);
Image3U RGB16ImageFromYUVOpsin(const Image3U& yuv, int bit_depth);
Image3F RGBLinearImageFromYUVOpsin(const Image3U& yuv, int bit_depth);

Image3U YUVOpsinImageFromRGB8(const Image3B& rgb, int out_bit_depth);
Image3U YUVOpsinImageFromRGB16(const Image3U& rgb, int out_bit_depth);
Image3U YUVOpsinImageFromRGBLinear(const Image3F& rgb, int out_bit_depth);

}  // namespace pik

#endif  // PIK_YUV_OPSIN_CONVERT_H_
