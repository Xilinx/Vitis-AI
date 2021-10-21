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

#include "xf_boundingbox_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPIX)) / 8) / (INPUT_PTR_WIDTH / 8);

void boundingbox_accel(
    ap_uint<INPUT_PTR_WIDTH>* in_img, int* roi, int color_info[MAX_BOXES][4], int height, int width, int num_box) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=in_img  	offset=slave bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=roi  	    	offset=slave bundle=gmem2 depth=20
    #pragma HLS INTERFACE m_axi     port=color_info  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
 
    #pragma HLS INTERFACE s_axilite port=roi            	 
    #pragma HLS INTERFACE s_axilite port=color_info          
    #pragma HLS INTERFACE s_axilite port=num_box           	 
    #pragma HLS INTERFACE s_axilite port=height              
    #pragma HLS INTERFACE s_axilite port=width               
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Rect_<int> _roi[MAX_BOXES];
    xf::cv::Scalar<4, unsigned char> color[MAX_BOXES];
    for (int i = 0, j = 0; i < num_box; i++, j += 4) {
        _roi[i].x = roi[j];
        _roi[i].y = roi[j + 1];
        _roi[i].height = roi[j + 2];
        _roi[i].width = roi[j + 3];
    }
    for (int i = 0; i < (num_box); i++) {
        for (int j = 0; j < XF_CHANNELS(TYPE, NPIX); j++) {
            color[i].val[j] = color_info[i][j]; //(i*XF_CHANNELS(TYPE,NPIX))+j];
        }
    }

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPIX> in_mat(height, width, in_img);
    xf::cv::boundingbox<TYPE, HEIGHT, WIDTH, MAX_BOXES, NPIX>(in_mat, _roi, color, num_box);
}
