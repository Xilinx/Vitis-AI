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

namespace xf {
namespace cv {
namespace aie {

//@Get functions {
inline metadata_elem_t xfGetTileWidth(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_TILEWIDTH];
}

inline metadata_elem_t xfGetTileHeight(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_TILEHEIGHT];
}

inline metadata_elem_t xfGetTilePosH(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_POSITIONH];
}

inline metadata_elem_t xfGetTilePosV(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_POSITIONV];
}

inline metadata_elem_t xfGetTileOVLP_HL(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPH_LEFT];
}

inline metadata_elem_t xfGetTileOVLP_HR(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPH_RIGHT];
}

inline metadata_elem_t xfGetTileOVLP_VT(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPV_TOP];
}

inline metadata_elem_t xfGetTileOVLP_VB(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPV_BOTTOM];
}

inline metadata_elem_t xfGetTileDatBitwidth(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_DATA_BITWIDTH];
}

inline metadata_elem_t xfGetTileFinalWidth(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_FINAL_WIDTH];
}

inline metadata_elem_t xfGetTileFinalHeight(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_FINAL_HEIGHT];
}

inline metadata_elem_t xfGetTileCrctPosH(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_CRCTPOSH];
}

inline metadata_elem_t xfGetTileCrctPosV(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_CRCTPOSV];
}

inline metadata_elem_t xfGetTileCrctTWidth(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_CRCT_TWIDTH];
}

inline metadata_elem_t xfGetTileCrctTHeight(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_CRCT_THEIGHT];
}

inline metadata_elem_t xfGetTileSat(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr)[POS_MDS_SAT_EN];
}

inline void* xfGetImgDataPtr(void* img_ptr) {
    return ((metadata_elem_t*)img_ptr + POS_MDS_IMG_PTR);
}
//@}

//@Set functions {
inline void xfSetTileWidth(void* img_ptr, metadata_elem_t width) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_TILEWIDTH] = width;
}

inline void xfSetTileHeight(void* img_ptr, metadata_elem_t height) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_TILEHEIGHT] = height;
}

inline void xfSetTilePosH(void* img_ptr, metadata_elem_t posH) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_POSITIONH] = posH;
}

inline void xfSetTilePosV(void* img_ptr, metadata_elem_t posV) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_POSITIONV] = posV;
}

inline void xfSetTileOVLP_HL(void* img_ptr, metadata_elem_t ovrlp_HL) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPH_LEFT] = ovrlp_HL;
}

inline void xfSetTileOVLP_HR(void* img_ptr, metadata_elem_t ovrlp_HR) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPH_RIGHT] = ovrlp_HR;
}

inline void xfSetTileOVLP_VT(void* img_ptr, metadata_elem_t ovrlp_VT) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPV_TOP] = ovrlp_VT;
}

inline void xfSetTileOVLP_VB(void* img_ptr, metadata_elem_t ovrlp_VB) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_OVLPV_BOTTOM] = ovrlp_VB;
}

inline void xfSetTileDatBitwidth(void* img_ptr, metadata_elem_t data_bitwidth) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_DATA_BITWIDTH] = data_bitwidth;
}

inline void xfSetTileFinalWidth(void* img_ptr, metadata_elem_t final_width) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_FINAL_WIDTH] = final_width;
}

inline void xfSetTileFinalHeight(void* img_ptr, metadata_elem_t final_height) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_FINAL_HEIGHT] = final_height;
}

inline void xfSetTileCrctPosH(void* img_ptr, metadata_elem_t crct_posh) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_CRCTPOSH] = crct_posh;
}

inline void xfSetTileCrctPosV(void* img_ptr, metadata_elem_t crct_posv) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_CRCTPOSV] = crct_posv;
}

inline void xfSetTileCrctTWidth(void* img_ptr, metadata_elem_t crct_twidth) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_CRCT_TWIDTH] = crct_twidth;
}

inline void xfSetTileCrctTHeight(void* img_ptr, metadata_elem_t crct_theight) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_CRCT_THEIGHT] = crct_theight;
}

inline void xfUnsignedSaturation(void* img_ptr) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_SAT_EN] = 1;
}

inline void xfDefaultSaturation(void* img_ptr) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_SAT_EN] = 0;
}

inline void xfSignedSaturation(void* img_ptr) {
    ((metadata_elem_t*)img_ptr)[POS_MDS_SAT_EN] = 2;
}
//@}
}
}
}
#endif
