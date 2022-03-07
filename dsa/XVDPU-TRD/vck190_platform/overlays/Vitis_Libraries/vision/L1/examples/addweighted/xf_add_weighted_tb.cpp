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
#include "xf_add_weighted_config.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path1> <input image path2> \n");
        return -1;
    }

    cv::Mat in_gray, in_gray1, ocv_ref, out_gray, diff, ocv_ref_in1, ocv_ref_in2, inout_gray1;

#if GRAY
    in_gray = cv::imread(argv[1], 0);  // read image1
    in_gray1 = cv::imread(argv[2], 0); // read image2
#else
    in_gray = cv::imread(argv[1], 1);  // read image1
    in_gray1 = cv::imread(argv[2], 1); // read image2

#endif
    if (in_gray.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[1]);
        return -1;
    }
    if (in_gray1.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[2]);
        return -1;
    }
    int height = in_gray.rows;
    int width = in_gray.cols;
#if GRAY
    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC1);
    out_gray.create(in_gray.rows, in_gray.cols, CV_8UC1);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC1);
#else
    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC3);
    out_gray.create(in_gray.rows, in_gray.cols, CV_8UC3);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC3);
#endif
    float alpha = 0.2;
    float beta = 0.8;
    float gama = 0.0;

    // OpenCV function
    cv::addWeighted(in_gray, alpha, in_gray1, beta, gama, ocv_ref);

    // Write OpenCV reference image
    cv::imwrite("out_ocv.jpg", ocv_ref);

    // Call the top function
    add_weighted_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray.data, (ap_uint<INPUT_PTR_WIDTH>*)in_gray1.data, alpha, beta,
                       gama, (ap_uint<OUTPUT_PTR_WIDTH>*)out_gray.data, height, width);

    imwrite("out_hls.jpg", out_gray);

    // Compute absolute difference image
    absdiff(out_gray, ocv_ref, diff);

    // Save the difference image
    imwrite("diff.png", diff);

    // Find minimum and maximum differences
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray.rows; i++) {
        for (int j = 0; j < in_gray.cols; j++) {
            unsigned char v = diff.at<unsigned char>(i, j);
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
