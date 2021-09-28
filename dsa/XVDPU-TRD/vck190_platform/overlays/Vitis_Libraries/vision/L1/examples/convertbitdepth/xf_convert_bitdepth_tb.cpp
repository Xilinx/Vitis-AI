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
#include "xf_convert_bitdepth_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img;
    cv::Mat in_gray, input_img; //, ocv_ref;

    // Reading in the image:
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Convert first to initial type:
    in_img.convertTo(input_img, OCV_INTYPE);

    // Create memory for output image
    cv::Mat ocv_ref(in_img.rows, in_img.cols, OCV_OUTTYPE);
    cv::Mat diff(in_img.rows, in_img.cols, OCV_OUTTYPE);
    cv::Mat out_img(in_img.rows, in_img.cols, OCV_OUTTYPE);

    // Opencv reference::
    input_img.convertTo(ocv_ref, OCV_OUTTYPE);
    cv::imwrite("out_ocv.jpg", ocv_ref);

    int shift = 0;
    int height = in_img.rows;
    int width = in_img.cols;

    convert_bitdepth_accel((ap_uint<INPUT_PTR_WIDTH>*)input_img.data, shift, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data,
                           height, width);

    // Write output image:
    cv::imwrite("hls_out.png", out_img);

    // Results verification:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("out_err.png", diff);

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
