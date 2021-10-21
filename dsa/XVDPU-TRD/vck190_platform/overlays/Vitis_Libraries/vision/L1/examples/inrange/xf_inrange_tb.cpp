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
#include "xf_inrange_config.h"
#include <ap_int.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1>\n");
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, in_gray, diff;

// Reading in the image:
#if RGB
    in_img = cv::imread(argv[1], 1);
#else
    in_img = cv::imread(argv[1], 0);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Create memory for outputs:
    ocv_ref.create(in_img.rows, in_img.cols, in_img.depth());
    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    diff.create(in_img.rows, in_img.cols, in_img.depth());

    // Reference function in OpenCV:
    unsigned char lower_thresh = 50;
    unsigned char upper_thresh = 100;
    int height = in_img.rows;
    int width = in_img.cols;

#if RGB
    cv::inRange(in_img, cv::Scalar(lower_thresh, lower_thresh, lower_thresh),
                cv::Scalar(upper_thresh, upper_thresh, upper_thresh), ocv_ref);
#else
    cv::inRange(in_img, cv::Scalar(lower_thresh), cv::Scalar(upper_thresh), ocv_ref);
#endif

    inrange_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, lower_thresh, upper_thresh,
                  (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, height, width);

    // Write output image:
    cv::imwrite("hls_out.jpg", out_img);
    cv::imwrite("ref_img.jpg", ocv_ref); // reference image

    // Results verification:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff_img.jpg", diff); // Save the difference image for debugging purpose

    // Find minimum and maximum differences:
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
#if RGB
            cv::Vec3b v = diff.at<cv::Vec3b>(i, j);
            if (v[0] > 1) cnt++;
            if (v[1] > 1) cnt++;
            if (v[2] > 1) cnt++;

            if (minval > v[0]) minval = v[0];
            if (minval > v[1]) minval = v[1];
            if (minval > v[2]) minval = v[2];

            if (maxval < v[0]) maxval = v[0];
            if (maxval < v[1]) maxval = v[1];
            if (maxval < v[2]) maxval = v[2];
#else
            uchar v = diff.at<uchar>(i, j);
            if (v > 1) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
#endif
        }
    }
    float err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols * in_img.channels());

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
