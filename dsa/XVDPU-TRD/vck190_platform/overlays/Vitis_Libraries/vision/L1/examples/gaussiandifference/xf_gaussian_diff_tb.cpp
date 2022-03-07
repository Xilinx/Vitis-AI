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
#include "xf_gaussian_diff_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat out_img, ocv_ref, in_gray, diff;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (!in_gray.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Create memory for output image
    ocv_ref.create(in_gray.rows, in_gray.cols, in_gray.depth());
    out_img.create(in_gray.rows, in_gray.cols, in_gray.depth());

#if FILTER_WIDTH == 3
    float sigma = 0.5f;
#endif
#if FILTER_WIDTH == 7
    float sigma = 1.16666f;
#endif
#if FILTER_WIDTH == 5
    float sigma = 0.8333f;
#endif

    // OpenCL section:
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;

    int rows = in_gray.rows;
    int cols = in_gray.cols;

    //////////Top function call //////////////////////////////
    gaussian_diff_accel((ap_uint<PTR_WIDTH>*)in_gray.data, sigma, (ap_uint<PTR_WIDTH>*)out_img.data, rows, cols);

    // Write the output of kernel:
    cv::imwrite("output_hls.png", out_img);

    return 0;
}
