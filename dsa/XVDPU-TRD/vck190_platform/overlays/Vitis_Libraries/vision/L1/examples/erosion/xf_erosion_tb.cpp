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
#include "xf_erosion_config.h"
#include <ap_int.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1>\n");
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_img1, out_img, ocv_ref;
    cv::Mat in_gray, in_gray1, diff;
#if GRAY
    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);
#else
    // Reading in the image:
    in_gray = cv::imread(argv[1], 1);
#endif
    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
    int height = in_gray.rows;
    int width = in_gray.cols;
#if GRAY
    // Create memory for output images:
    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC1);
    out_img.create(in_gray.rows, in_gray.cols, CV_8UC1);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC1);
#else
    // Create memory for output images:
    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC3);
    out_img.create(in_gray.rows, in_gray.cols, CV_8UC3);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC3);
#endif
    // OpenCV reference:
    cv::Mat element = cv::getStructuringElement(KERNEL_SHAPE, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));
    cv::erode(in_gray, ocv_ref, element, cv::Point(-1, -1), ITERATIONS, cv::BORDER_CONSTANT);
    cv::imwrite("out_ocv.jpg", ocv_ref);

    // HLS implementation:
    unsigned char shape[FILTER_SIZE * FILTER_SIZE];

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        shape[i] = element.data[i];
    }
    // Write output image
    cv::imwrite("hls_out.jpg", out_img);

    // Call the top function
    erosion_accel((ap_uint<PTR_WIDTH>*)in_gray.data, shape, (ap_uint<PTR_WIDTH>*)out_img.data, height, width);

    //  Compute absolute difference:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("out_error.jpg", diff);

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray.rows; i++) {
        for (int j = 0; j < in_gray.cols; j++) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 0) cnt++;
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
