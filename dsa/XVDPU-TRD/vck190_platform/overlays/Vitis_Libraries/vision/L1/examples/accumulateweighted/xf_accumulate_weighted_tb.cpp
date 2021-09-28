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
#include <stdlib.h>
#include <ap_int.h>
#include "xf_accumulate_weighted_config.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n");
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_img1, out_img;
    cv::Mat in_gray, in_gray1, diff;
#if GRAY
    in_gray = cv::imread(argv[1], 0);  // read image
    in_gray1 = cv::imread(argv[2], 0); // read image
#else
    in_gray = cv::imread(argv[1], 1);  // read image
    in_gray1 = cv::imread(argv[2], 1); // read image
#endif
    int height = in_gray.rows;
    int width = in_gray.cols;
    if (in_gray.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[1]);
        return -1;
    }
    if (in_gray1.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[2]);
        return -1;
    }
#if GRAY
    cv::Mat inout_gray(in_gray.rows, in_gray.cols, CV_16U, 1);
    cv::Mat out_gray(in_gray.rows, in_gray.cols, CV_16U, 1);
    cv::Mat inout_gray1(in_gray.rows, in_gray.cols, CV_32FC1, 1);

    cv::Mat ocv_ref(in_gray.rows, in_gray.cols, CV_16U, 1);
    cv::Mat ocv_ref_in1(in_gray.rows, in_gray.cols, CV_32FC1, 1);
    cv::Mat ocv_ref_in2(in_gray.rows, in_gray.cols, CV_32FC1, 1);

    in_gray.convertTo(ocv_ref_in1, CV_32FC1);
    in_gray1.convertTo(ocv_ref_in2, CV_32FC1);
#else
    cv::Mat inout_gray(in_gray.rows, in_gray.cols, CV_16UC3, 1);
    cv::Mat out_gray(in_gray.rows, in_gray.cols, CV_16UC3, 1);
    cv::Mat inout_gray1(in_gray.rows, in_gray.cols, CV_32FC3, 1);

    cv::Mat ocv_ref(in_gray.rows, in_gray.cols, CV_16UC3, 1);
    cv::Mat ocv_ref_in1(in_gray.rows, in_gray.cols, CV_32FC3, 1);
    cv::Mat ocv_ref_in2(in_gray.rows, in_gray.cols, CV_32FC3, 1);

    in_gray.convertTo(ocv_ref_in1, CV_32FC3);
    in_gray1.convertTo(ocv_ref_in2, CV_32FC3);

#endif
    // Weight ( 0 to 1 )
    float alpha = 0.76;

    // OpenCV function
    cv::accumulateWeighted(ocv_ref_in1, ocv_ref_in2, alpha, cv::noArray());
#if GRAY
    ocv_ref_in2.convertTo(ocv_ref, CV_16U);
#else
    ocv_ref_in2.convertTo(ocv_ref, CV_16UC3);
#endif
    // Write OpenCV reference image
    cv::imwrite("out_ocv.jpg", ocv_ref);

    accumulate_weighted_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray.data, (ap_uint<INPUT_PTR_WIDTH>*)in_gray1.data,
                              (ap_uint<OUTPUT_PTR_WIDTH>*)out_gray.data, alpha, height, width);

    cv::imwrite("out_hls.jpg", out_gray);
#if GRAY
    out_gray.convertTo(inout_gray1, CV_32FC1);
#else
    out_gray.convertTo(inout_gray1, CV_32FC3);
#endif

    // Compute absolute difference image
    absdiff(ocv_ref_in2, inout_gray1, diff);

    // Save the difference image
    cv::imwrite("diff.png", diff);

    // Find minimum and maximum differences
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray.rows; i++) {
        for (int j = 0; j < in_gray.cols; j++) {
            float v = diff.at<float>(i, j);
            if (v > 1) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    float err_per = 100.0 * (float)cnt / (in_gray.rows * in_gray.cols);

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
