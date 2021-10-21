// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_YUV_CONVERT_H_
#define PIK_YUV_CONVERT_H_

#include "pik/image.h"

namespace pik {

Image3B RGB8ImageFromYUVRec709(const Image3U& yuv, int bit_depth);
Image3U RGB16ImageFromYUVRec709(const Image3U& yuv, int bit_depth);
Image3F RGBLinearImageFromYUVRec709(const Image3U& yuv, int bit_depth);

Image3U YUVRec709ImageFromRGB8(const Image3B& rgb, int out_bit_depth);
Image3U YUVRec709ImageFromRGB16(const Image3U& rgb, int out_bit_depth);
Image3U YUVRec709ImageFromRGBLinear(const Image3F& rgb, int out_bit_depth);

void SubSampleChroma(const Image3U& yuv, int bit_depth, ImageU* yplane,
                     ImageU* uplane, ImageU* vplane);

Image3U SuperSampleChroma(const ImageU& yplane, const ImageU& uplane,
                          const ImageU& vplane, int bit_depth);

}  // namespace pik

#endif  // PIK_YUV_CONVERT_H_
