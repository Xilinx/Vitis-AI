/*
 * Copyright 2019 Xilinx, Inc.
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
#include "xf_autowhitebalance_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);

static bool flag;

static uint32_t hist0[3][HIST_SIZE];
static uint32_t hist1[3][HIST_SIZE];
static int igain_0[3];
static int igain_1[3];

void AWBKernel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
               ap_uint<OUTPUT_PTR_WIDTH>* img_out,
               int height,
               int width,
               uint32_t hist0[3][HIST_SIZE],
               uint32_t hist1[3][HIST_SIZE],
               int gain0[3],
               int gain1[3],
               float thresh,
               float inputMin,
               float inputMax,
               float outputMin,
               float outputMax) {
#pragma HLS INLINE OFF

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> in_mat(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> out_mat(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> impop(height, width);

// clang-format off
#pragma HLS DATAFLOW
// clang-format on

// clang-format off
#pragma HLS stream variable=impop.data dim=1 depth=2
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_inp, in_mat);

    if (WB_TYPE == 1) {
        xf::cv::AWBhistogram<IN_TYPE, IN_TYPE, HEIGHT, WIDTH, NPC1, 1>(in_mat, impop, hist0, thresh, inputMin, inputMax,
                                                                       outputMin, outputMax);
        xf::cv::AWBNormalization<IN_TYPE, IN_TYPE, HEIGHT, WIDTH, NPC1, 1>(impop, out_mat, hist1, thresh, inputMin,
                                                                           inputMax, outputMin, outputMax);
    } else {
        xf::cv::AWBChannelGain<IN_TYPE, IN_TYPE, HEIGHT, WIDTH, NPC1, 0>(in_mat, impop, thresh, gain0);
        xf::cv::AWBGainUpdate<IN_TYPE, IN_TYPE, HEIGHT, WIDTH, NPC1, 0>(impop, out_mat, thresh, gain1);
    }
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(out_mat, img_out);
}
extern "C" {
void autowhitebalance_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                            ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                            float thresh,
                            int rows,
                            int cols,
                            float inputMin,
                            float inputMax,
                            float outputMin,
                            float outputMax) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 depth=__XF_DEPTH
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem3  depth=__XF_DEPTH

#pragma HLS INTERFACE s_axilite port=thresh   
#pragma HLS INTERFACE s_axilite port=inputMin 
#pragma HLS INTERFACE s_axilite port=inputMax 
#pragma HLS INTERFACE s_axilite port=outputMin
#pragma HLS INTERFACE s_axilite port=outputMax
#pragma HLS INTERFACE s_axilite port=rows     
#pragma HLS INTERFACE s_axilite port=cols     
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS ARRAY_PARTITION variable=hist0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=hist1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=igain_0 complete 
#pragma HLS ARRAY_PARTITION variable=igain_1 complete
    // clang-format on

    if (!flag) {
        AWBKernel(img_inp, img_out, rows, cols, hist0, hist1, igain_0, igain_1, thresh, inputMin, inputMax, outputMin,
                  outputMax);

        flag = 1;

    } else {
        AWBKernel(img_inp, img_out, rows, cols, hist1, hist0, igain_1, igain_0, thresh, inputMin, inputMax, outputMin,
                  outputMax);

        flag = 0;
    }
}
}