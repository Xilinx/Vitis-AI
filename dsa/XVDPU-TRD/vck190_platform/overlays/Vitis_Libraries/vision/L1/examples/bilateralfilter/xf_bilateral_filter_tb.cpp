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
#include "xf_bilateral_filter_config.h"

int main(int argc, char** argv) {
    cv::Mat in_img, out_img, ocv_ref, in_img_gau;
    cv::Mat in_gray, in_gray1, diff;

    cv::RNG rng;

    uchar error_threshold = 0;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

#if GRAY
    in_img = cv::imread(argv[1], 0); // reading in the gray image
#else
    in_img = cv::imread(argv[1], 1); // reading in the color image
#endif

    if (!in_img.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

// create memory for output image
#if GRAY
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1);
    out_img.create(in_img.rows, in_img.cols, CV_8UC1); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC1);
#else
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img.create(in_img.rows, in_img.cols, CV_8UC3); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
#endif

    float sigma_color = rng.uniform(0.0, 1.0) * 255;
    float sigma_space = rng.uniform(0.0, 1.0);

    std::cout << " sigma_color: " << sigma_color << " sigma_space: " << sigma_space << std::endl;

    // OpenCV bilateral filter function
    cv::bilateralFilter(in_img, ocv_ref, FILTER_WIDTH, sigma_color, sigma_space, cv::BORDER_REPLICATE);

    cv::imwrite("output_ocv.png", ocv_ref);

// OpenCL section:
#if GRAY
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 1 * sizeof(unsigned char);
#else
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
#endif
    size_t image_out_size_bytes = image_in_size_bytes;

    // Call the top function
    bilateral_filter_accel((ap_uint<PTR_WIDTH>*)in_img.data, sigma_color, sigma_space, in_img.rows, in_img.cols,
                           (ap_uint<PTR_WIDTH>*)out_img.data);

    // Write output image
    cv::imwrite("hls_out.jpg", out_img);

    // Compute absolute difference image
    cv::absdiff(ocv_ref, out_img, diff);
    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 10, err_per);

    if (err_per > 0.0f) {
        return 1;
    }

    return 0;
}
