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

#ifndef _XF_AIE_CONST_H_
#define _XF_AIE_CONST_H_
#include <stdint.h>

namespace xf {
namespace cv {
namespace aie {

static constexpr int METADATA_ELEMENTS = 32;
using metadata_elem_t = int16_t;
static constexpr int METADATA_SIZE = METADATA_ELEMENTS * sizeof(metadata_elem_t);

enum SmartTileMDPOS {
    POS_MDS_IN_OFFSET = 0,
    POS_MDS_TILEWIDTH = 2,
    POS_MDS_TILEHEIGHT = 3,
    POS_MDS_OVLPH_LEFT = 4,
    POS_MDS_OVLPH_RIGHT = 5,
    POS_MDS_OVLPV_TOP = 6,
    POS_MDS_OVLPV_BOTTOM = 7,
    POS_MDS_OUT_OFFSET = 8,
    POS_MDS_CRCT_TWIDTH = 10,
    POS_MDS_CRCT_THEIGHT = 11,
    POS_MDS_SAT_EN = 12,
    POS_MDS_CRCTPOSH = 13,
    POS_MDS_CRCTPOSV = 14,
    POS_MDS_POSITIONH = 15,
    POS_MDS_POSITIONV = 16,
    POS_MDS_DATA_BITWIDTH = 17,
    POS_MDS_FINAL_WIDTH = 18,
    POS_MDS_FINAL_HEIGHT = 19,
    POS_MDS_IMG_PTR = 32
    };
}
}
}

#endif
