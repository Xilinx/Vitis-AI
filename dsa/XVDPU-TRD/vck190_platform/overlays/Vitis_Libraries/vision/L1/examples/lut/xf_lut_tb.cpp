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
#include "xf_lut_config.h"
#include <ap_int.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1>\n");
        return EXIT_FAILURE;
    }

    // Gamma correction LUT for example:
    //    std::vector<unsigned char>
    unsigned char lut[256] = {
        0,   16,  23,  28,  32,  36,  39,  42,  45,  48,  50,  53,  55,  58,  60,  62,  64,  66,  68,  70,  71,  73,
        75,  77,  78,  80,  81,  83,  84,  86,  87,  89,  90,  92,  93,  94,  96,  97,  98,  100, 101, 102, 103, 105,
        106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129,
        130, 131, 132, 133, 134, 135, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 145, 146, 147, 148, 149,
        150, 151, 151, 152, 153, 154, 155, 156, 156, 157, 158, 159, 160, 160, 161, 162, 163, 164, 164, 165, 166, 167,
        167, 168, 169, 170, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 179, 179, 180, 181, 181, 182, 183,
        183, 184, 185, 186, 186, 187, 188, 188, 189, 190, 190, 191, 192, 192, 193, 194, 194, 195, 196, 196, 197, 198,
        198, 199, 199, 200, 201, 201, 202, 203, 203, 204, 204, 205, 206, 206, 207, 208, 208, 209, 209, 210, 211, 211,
        212, 212, 213, 214, 214, 215, 215, 216, 217, 217, 218, 218, 219, 220, 220, 221, 221, 222, 222, 223, 224, 224,
        225, 225, 226, 226, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232, 233, 233, 234, 234, 235, 235, 236, 236,
        237, 237, 238, 238, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 248,
        248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255};

    cv::Mat in_img, in_gray, out_img, ocv_ref, diff, lut_mat;

// Reading in the color image:
#if GRAY
    in_img = cv::imread(argv[1], 0);
#else
    in_img = cv::imread(argv[1], 1);
#endif
    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    int height = in_img.rows;
    int width = in_img.cols;

// Memory allocation for the output images:
#if GRAY
    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    ocv_ref.create(in_img.rows, in_img.cols, in_img.depth());
    diff.create(in_img.rows, in_img.cols, in_img.depth());
#else
    out_img.create(in_img.rows, in_img.cols, CV_8UC3);
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3);
    diff.create(in_img.rows, in_img.cols, CV_8UC3);

#endif

    // Call the top function
    lut_accel((ap_uint<PTR_WIDTH>*)in_img.data, lut, (ap_uint<PTR_WIDTH>*)out_img.data, height, width);

    // Write output image
    cv::imwrite("hls_out.jpg", out_img);

    // OpenCV reference:
    lut_mat = cv::Mat(1, 256, CV_8UC1, lut);
    cv::LUT(in_img, lut_mat, ocv_ref);
    cv::imwrite("ref_img.jpg", ocv_ref);

    // Results verification:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff_img.jpg", diff);

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
