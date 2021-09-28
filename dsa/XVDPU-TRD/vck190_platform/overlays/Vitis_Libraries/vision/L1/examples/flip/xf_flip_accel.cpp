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

#include "xf_flip_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (PTR_WIDTH / 8);

void flip_accel(ap_uint<PTR_WIDTH>* SrcPtr, ap_uint<PTR_WIDTH>* DstPtr, int Rows, int Cols, int Direction) {
// clang-format off
#pragma HLS INTERFACE m_axi port=SrcPtr offset=slave bundle=gmem0 depth=__XF_DEPTH
#pragma HLS INTERFACE m_axi port=DstPtr offset=slave bundle=gmem1 depth=__XF_DEPTH
#pragma HLS INTERFACE s_axilite port=Rows
#pragma HLS INTERFACE s_axilite port=Cols
#pragma HLS INTERFACE s_axilite port=Direction
#pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::flip<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(SrcPtr, DstPtr, Rows, Cols, Direction);

    return;
} // End of kernel
