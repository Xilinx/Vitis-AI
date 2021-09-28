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

#include "xf_crop_config.h"

extern "C" {
void crop_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                ap_uint<OUTPUT_PTR_WIDTH>* _dst,
                ap_uint<OUTPUT_PTR_WIDTH>* _dst1,
                ap_uint<OUTPUT_PTR_WIDTH>* _dst2,
                int* roi,
                int height,
                int width)
//	void crop_accel(ap_uint<INPUT_PTR_WIDTH> *img_in, ap_uint<OUTPUT_PTR_WIDTH> *_dst,int *roi, int height, int
// width)
{
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi     port=_dst  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=_dst1  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=_dst2  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=roi   offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=height     
    #pragma HLS INTERFACE s_axilite port=width     
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    printf("started loading rect execution\n");
    xf::cv::Rect_<unsigned int> _roi[NUM_ROI];
    for (int i = 0, j = 0; j < (NUM_ROI * 4); i++, j += 4) {
        _roi[i].x = roi[j];
        _roi[i].y = roi[j + 1];
        _roi[i].height = roi[j + 2];
        _roi[i].width = roi[j + 3];
    }

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> in_mat(height, width, img_in);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> out_mat(_roi[0].height, _roi[0].width, _dst);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> out_mat1(_roi[1].height, _roi[1].width, _dst1);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> out_mat2(_roi[2].height, _roi[2].width, _dst2);

    xf::cv::crop<TYPE, HEIGHT, WIDTH, MEMORYMAPPED_ARCH, NPC>(in_mat, out_mat, _roi[0]);
    xf::cv::crop<TYPE, HEIGHT, WIDTH, MEMORYMAPPED_ARCH, NPC>(in_mat, out_mat1, _roi[1]);
    xf::cv::crop<TYPE, HEIGHT, WIDTH, MEMORYMAPPED_ARCH, NPC>(in_mat, out_mat2, _roi[2]);
}
}
