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
#include "xf_dilation_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_ref;
    cv::Mat diff;

// reading in the image
#if GRAY
    in_img = cv::imread(argv[1], 0);
#else
    in_img = cv::imread(argv[1], 1);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    int height = in_img.rows;
    int width = in_img.cols;
// create memory for output images
#if GRAY
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1);
    out_img.create(in_img.rows, in_img.cols, CV_8UC1);
    diff.create(in_img.rows, in_img.cols, CV_8UC1);

#else
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img.create(in_img.rows, in_img.cols, CV_8UC3);
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
#endif

    cv::Mat element = cv::getStructuringElement(KERNEL_SHAPE, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));
    cv::dilate(in_img, ocv_ref, element, cv::Point(-1, -1), ITERATIONS, cv::BORDER_CONSTANT);
    cv::imwrite("out_ocv.jpg", ocv_ref);
    /////////////////////	End of OpenCV reference	 ////////////////
    ////////////////////	HLS TOP function call	/////////////////

    unsigned char structure_element[FILTER_SIZE * FILTER_SIZE];

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        structure_element[i] = element.data[i];
    }
    // Write output image
    cv::imwrite("hw_out.jpg", out_img);

    dilation_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, structure_element,
                   height, width);

    //////////////////  Compute Absolute Difference ////////////////////
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("out_error.jpg", diff);

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 0) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    float err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) return 1;

    return 0;
}
