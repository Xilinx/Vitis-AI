/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _XF_AIE_UTILS_H_
#define _XF_AIE_UTILS_H_

#include <common/xf_aie_const.hpp>

inline int16_t xfcvGetTileWidth(int16_t* img_ptr) {
    return *img_ptr;
}

inline int16_t xfcvGetTileHeight(int16_t* img_ptr) {
    return *(img_ptr + 4);
}

inline void xfcvSetTileWidth(int16_t* img_ptr, int width) {
    *img_ptr = width;
}

inline void xfcvSetTileHeight(int16_t* img_ptr, int height) {
    *(img_ptr + 4) = height;
}

inline void xfcvSetTilePosH(int16_t* img_ptr, int div_factor) {
    int posH = *(img_ptr + 8);
    *(img_ptr + 8) = posH / div_factor;
}

inline void xfcvSetTilePosV(int16_t* img_ptr, int div_factor) {
    int posV = *(img_ptr + 12);
    *(img_ptr + 12) = posV / div_factor;
}

void xfcvSetUVMetaData(int16_t* img_ptr) {
    *img_ptr = *(img_ptr) / 2;             // width
    *(img_ptr + 4) = *(img_ptr + 4) / 2;   // height
    *(img_ptr + 8) = *(img_ptr + 8) / 2;   // posh
    *(img_ptr + 12) = *(img_ptr + 12) / 2; // posv
    *(img_ptr + 44) = *(img_ptr + 44) / 2; // crctposh
    *(img_ptr + 48) = *(img_ptr + 48) / 2; // crctposv
    *(img_ptr + 52) = *(img_ptr + 52) / 2; // crcttw
    *(img_ptr + 56) = *(img_ptr + 56) / 2; // crctth
}

inline void xfcvCopyMetaData(int16_t* img_in_ptr, int16_t* img_out_ptr) {
    v64int16* restrict in = (v64int16*)img_in_ptr;
    v64int16* restrict out = (v64int16*)img_out_ptr;
    *(out) = *(in);
    return;
}

inline int16_t* xfcvGetImgDataPtr(int16_t* img_ptr) {
    return (img_ptr + SMARTTILE_ELEMENTS);
}

inline void xfcvUnsignedSaturation(int16_t* img_ptr) {
    img_ptr[SMARTTILE_ELEMENTS - 4] = 1;
}

inline void xfcvSignedSaturation(int16_t* img_ptr) {
    img_ptr[SMARTTILE_ELEMENTS - 4] = 2;
}

#endif
