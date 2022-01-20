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

static constexpr int METADATA_ELEMENTS = 64;
using metadata_elem_t = int16_t;
static constexpr int METADATA_SIZE = METADATA_ELEMENTS * sizeof(metadata_elem_t);

enum SmartTileMDPOS {
    POS_MDS_TILEWIDTH = 0,
    POS_MDS_TILEHEIGHT = 4,
    POS_MDS_POSITIONH = 8,
    POS_MDS_POSITIONV = 12,
    POS_MDS_OVLPH_LEFT = 16,
    POS_MDS_OVLPH_RIGHT = 20,
    POS_MDS_OVLPV_TOP = 24,
    POS_MDS_OVLPV_BOTTOM = 28,
    POS_MDS_DATA_BITWIDTH = 32,
    POS_MDS_FINAL_WIDTH = 36,
    POS_MDS_FINAL_HEIGHT = 40,
    POS_MDS_CRCTPOSH = 44,
    POS_MDS_CRCTPOSV = 48,
    POS_MDS_CRCT_TWIDTH = 52,
    POS_MDS_CRCT_THEIGHT = 56,
    POS_MDS_SAT_EN = 60,
    POS_MDS_IMG_PTR = 64
};
}
}
}

#endif
