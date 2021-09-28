/*
 * Copyright 2020 Xilinx, Inc.
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

#include "xf_clahe_config.h"

#define CLAHE_T \
    xf::cv::clahe::CLAHEImpl<IN_TYPE, HEIGHT, WIDTH, NPC, CLIPLIMIT, TILES_Y_MAX, TILES_X_MAX, TILES_Y_MIN, TILES_X_MIN>

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * XF_PIXELWIDTH(IN_TYPE, NPC)) / PTR_WIDTH;
static constexpr int HIST_COUNTER_BITS = CLAHE_T::HIST_COUNTER_BITS;
static constexpr int CLIP_COUNTER_BITS = CLAHE_T::CLIP_COUNTER_BITS;

static bool flag = false;
static ap_uint<HIST_COUNTER_BITS> _lut1[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                       [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)];
static ap_uint<HIST_COUNTER_BITS> _lut2[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                       [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)];
static ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX];

void clahe_accel_i(ap_uint<PTR_WIDTH>* in_ptr,
                   ap_uint<PTR_WIDTH>* out_ptr,
                   ap_uint<HIST_COUNTER_BITS> _lutw[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                   [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                   ap_uint<HIST_COUNTER_BITS> _lutr[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                   [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                   ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX],
                   int height,
                   int width,
                   int clip,
                   int tilesY,
                   int tilesX) {
// clang-format off
#pragma HLS inline off
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> imgInput(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> imgOutput(height, width);
    CLAHE_T obj;

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC>(in_ptr, imgInput);
    obj.process(imgOutput, imgInput, _lutw, _lutr, _clipCounter, height, width, clip, tilesY, tilesX);
    xf::cv::xfMat2Array<PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC>(imgOutput, out_ptr);
    return;
}

void clahe_accel(
    ap_uint<PTR_WIDTH>* in_ptr, ap_uint<PTR_WIDTH>* out_ptr, int height, int width, int clip, int tilesY, int tilesX) {
// clang-format off
#pragma HLS INTERFACE m_axi      port=in_ptr  offset=slave bundle=gmem_in  depth=__XF_DEPTH
#pragma HLS INTERFACE m_axi      port=out_ptr offset=slave bundle=gmem_out depth=__XF_DEPTH
#pragma HLS INTERFACE s_axilite  port=height
#pragma HLS INTERFACE s_axilite  port=width
#pragma HLS INTERFACE s_axilite  port=clip
#pragma HLS INTERFACE s_axilite  port=tilesY
#pragma HLS INTERFACE s_axilite  port=tilesX
#pragma HLS INTERFACE s_axilite  port=return

#pragma HLS ARRAY_PARTITION variable=_lut1 dim=3 complete
#pragma HLS ARRAY_PARTITION variable=_lut2 dim=3 complete
    // clang-format on

    if (flag == false) {
        clahe_accel_i(in_ptr, out_ptr, _lut1, _lut2, _clipCounter, height, width, clip, tilesX, tilesY);
        flag = true;
    } else {
        clahe_accel_i(in_ptr, out_ptr, _lut2, _lut1, _clipCounter, height, width, clip, tilesX, tilesY);
        flag = false;
    }
}
