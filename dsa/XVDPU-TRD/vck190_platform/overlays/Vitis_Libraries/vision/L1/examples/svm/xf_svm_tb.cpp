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

#include "common/xf_headers.hpp"
#include "xf_svm_config.h"

/*****************************************************************************
 * 	 main function: SVM core
 *****************************************************************************/
int main() {
    float in_1[IN_ARRAY_SIZE_1], in_2[IN_ARRAY_SIZE_2];

    float a = 0, bias = 0.567;

    std::vector<float> v1, v2;

    // Input data feed
    for (int i = 0; i < TOTAL_ARRAY_ELEMENTS; i++) {
        in_1[i] = a;
        in_2[i] = 0.2 - a;

        if (in_1[i] >= 1) in_1[i] = 0.99;
        if (in_2[i] >= 1) in_2[i] = 0.99;

        a += 0.0008;
    }

    unsigned char out_frac;

    // top function call (fixed point computation)
    uint16_t ind1 = INDEX_ARR_1;
    uint16_t ind2 = INDEX_ARR_2;
    uint16_t arr_size1 = IN_ARRAY_SIZE_1;
    uint16_t arr_size2 = IN_ARRAY_SIZE_2;
    uint16_t frac1 = IN_FRAC_BITS_1;
    uint16_t frac2 = IN_FRAC_BITS_2;
    uint16_t n = NO_OF_KERNEL_ELEMENTS;

    // fixed point conversion of input data & filling into Mat
    ap_uint<16>* infix_1 = (ap_uint<16>*)malloc(IN_ARRAY_SIZE_1 * sizeof(short));
    ap_uint<16>* infix_2 = (ap_uint<16>*)malloc(IN_ARRAY_SIZE_1 * sizeof(short));

    memset(infix_1, 0, IN_ARRAY_SIZE_1 * sizeof(short));
    memset(infix_2, 0, IN_ARRAY_SIZE_1 * sizeof(short));

    for (int i = 0; i < TOTAL_ARRAY_ELEMENTS; i++) {
        infix_1[i] = (in_1[i] * pow(2, IN_FRAC_BITS_1));
        infix_2[i] = (in_2[i] * pow(2, IN_FRAC_BITS_2));
    }

    ap_int<64> resultFIX;
    uint16_t params[5] = {INDEX_ARR_1, INDEX_ARR_2, IN_FRAC_BITS_1, IN_FRAC_BITS_2, NO_OF_KERNEL_ELEMENTS};

    int bias_fix = bias * pow(2, out_frac);

    // HLS function
    svm_accel(infix_1, infix_2, params, &out_frac, &resultFIX);
    long int resultFIX_bias = resultFIX + bias_fix;
    float resultFl = (double)resultFIX_bias / pow(2, out_frac);

    // OpenCV reference function
    cv::Mat A(cv::Size(1, IN_ARRAY_SIZE_1), CV_16S);
    std::memcpy(A.data, infix_1, IN_ARRAY_SIZE_1 * sizeof(ap_int<16>));
    cv::Mat B(cv::Size(1, IN_ARRAY_SIZE_1), CV_16S);
    std::memcpy(B.data, infix_2, IN_ARRAY_SIZE_1 * sizeof(ap_int<16>));
    long int ocv_fix = (A.dot(B));
    long int ocv_fix_bias = (ocv_fix + bias_fix);
    float ocv_fl = (double)ocv_fix_bias / cv::pow(2, out_frac);

    std::cout << "hls fix: " << (long int)resultFIX << std::endl;
    std::cout << "ocv fix: " << ocv_fix << std::endl;

    std::cout << "HLS float: " << resultFl << std::endl;
    std::cout << "OCV float: " << ocv_fl << std::endl;

    // Error computation
    float diff = abs(abs(resultFl) - abs(ocv_fl));
    // std::cout << "HLS fixed point output: " << resultFIX << std::endl;

    std::cout << "Absolute difference: " << diff << std::endl;

    if (diff > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    std::cout << "Test Passed .... !!!" << std::endl;

    return 0;
}
