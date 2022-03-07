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
#include "xf_median_blur_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, diff;

//  Reading in the image:
#if GRAY
    in_img = cv::imread(argv[1], 0); // reading in the gray image
#else
    in_img = cv::imread(argv[1], 1); // reading in the color image
#endif

    if (in_img.data == NULL) {
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

    // OpenCV reference:
    cv::medianBlur(in_img, ocv_ref, WINDOW_SIZE);

// OpenCL section:
#if GRAY
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 1 * sizeof(unsigned char);
#else
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
#endif
    size_t image_out_size_bytes = image_in_size_bytes;

    // Call the top function
    median_blur_accel((ap_uint<PTR_WIDTH>*)in_img.data, in_img.rows, in_img.cols, (ap_uint<PTR_WIDTH>*)out_img.data);

    // Write down output images:
    cv::imwrite("hls_out.jpg", out_img); // kernel output
    cv::imwrite("ref_img.jpg", ocv_ref); // reference image

    absdiff(ocv_ref, out_img, diff);
    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 10, err_per);

    return 0;
}
