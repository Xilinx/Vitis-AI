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

#include "xf_svm_config.h"

extern "C" {

void svm_accel(ap_uint<PTR_IN_WIDTH>* img_in1,
               ap_uint<PTR_IN_WIDTH>* img_in2,
               unsigned short* params,
               unsigned char* fractional_out,
               ap_int<64>* result_out) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in1        offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=img_in2        offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=params         offset=slave  bundle=gmem2
    #pragma HLS INTERFACE m_axi      port=fractional_out offset=slave  bundle=gmem3
    #pragma HLS INTERFACE m_axi      port=result_out     offset=slave  bundle=gmem4
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, IN_ARRAY_SIZE_1, IN_ARRAY_SIZE_1, NPC1> imgInput1;
    xf::cv::Mat<IN_TYPE, IN_ARRAY_SIZE_2, IN_ARRAY_SIZE_2, NPC1> imgInput2;

    // Retrieve all the params:
    unsigned short index1 = params[0];
    unsigned short index2 = params[1];
    unsigned short frac1 = params[2];
    unsigned short frac2 = params[3];
    unsigned short n = params[4];

#pragma HLS DATAFLOW

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, IN_ARRAY_SIZE_1, IN_ARRAY_SIZE_1, NPC1>(img_in1, imgInput1);
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, IN_ARRAY_SIZE_2, IN_ARRAY_SIZE_2, NPC1>(img_in2, imgInput2);

    // Run xfOpenCV kernel:
    xf::cv::SVM<IN_TYPE, IN_TYPE, PTR_OUT_WIDTH, IN_ARRAY_SIZE_1, IN_ARRAY_SIZE_1, IN_ARRAY_SIZE_2, IN_ARRAY_SIZE_2,
                NPC1, NO_OF_KERNEL_ELEMENTS>(imgInput1, imgInput2, index1, index2, frac1, frac2, n, fractional_out,
                                             result_out);

    return;
} // End of kernel

} // End of extern C
