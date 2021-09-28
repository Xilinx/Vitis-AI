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
#include "xf_convertscaleabs_config.h"
#include <ap_int.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_gray, ocv_ref, out_gray, diff;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
    int height = in_gray.rows;
    int width = in_gray.cols;

    // Allocate memory for the outputs:
    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC1);
    out_gray.create(in_gray.rows, in_gray.cols, CV_8UC1);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC1);

    float scale = 1.5;
    float shift = 0;

    // Reference OpenCV function:
    cv::convertScaleAbs(in_gray, ocv_ref, scale, shift);

    // Write OpenCV reference image:
    cv::imwrite("out_ocv.jpg", ocv_ref);

    convertScaleAbs_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray.data, scale, shift,
                          (ap_uint<OUTPUT_PTR_WIDTH>*)out_gray.data, height, width);

    // Write kernel output:
    cv::imwrite("out_hls.jpg", out_gray);

    // Compute absolute difference image
    cv::absdiff(out_gray, ocv_ref, diff);

    // Save the difference image
    cv::imwrite("diff.png", diff);

    // Find minimum and maximum differences
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray.rows; i++) {
        for (int j = 0; j < in_gray.cols; j++) {
            char v = diff.at<unsigned char>(i, j);
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
