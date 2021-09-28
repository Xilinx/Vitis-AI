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
#include "xf_paintmask_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, in_gray, diff, in_mask;

    // Reading in the gray image:
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    int height = in_img.rows;
    int width = in_img.cols;
    // Allocate memory for output images:
    ocv_ref.create(in_img.rows, in_img.cols, in_img.depth());
    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    diff.create(in_img.rows, in_img.cols, in_img.depth());
    in_mask.create(in_img.rows, in_img.cols, CV_8UC1);

    uint64_t q = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            for (int c = 0; c < in_img.channels(); c++) {
                if ((j > 250) && (j < 750)) {
                    in_mask.data[q + c] = 255;
                } else {
                    in_mask.data[q + c] = 0;
                }
            }
            q += in_img.channels();
        }
    }

    unsigned char color[XF_CHANNELS(TYPE, NPC1)];
    for (int i = 0; i < in_img.channels(); i++) {
        color[i] = 150;
    }

    // Call the top function

    paintmask_accel((ap_uint<PTR_WIDTH>*)in_img.data, (ap_uint<PTR_WIDTH>*)in_mask.data, color,
                    (ap_uint<PTR_WIDTH>*)out_img.data, height, width);

    // Reference function:
    unsigned long long int p = 0;
    for (int i = 0; i < ocv_ref.rows; i++) {
        for (int j = 0; j < ocv_ref.cols; j++) {
            for (int c = 0; c < ocv_ref.channels(); c++) {
                if (in_mask.data[p + c] != 0) {
                    ocv_ref.data[p + c] = color[c];
                } else {
                    ocv_ref.data[p + c] = in_img.data[p + c];
                }
            }
            p += in_img.channels();
        }
    }

    // Write output image
    cv::imwrite("hls_out.jpg", out_img);
    cv::imwrite("ref_img.jpg", ocv_ref); // reference image

    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff_img.jpg", diff); // Save the difference image for debugging purpose

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 1) cnt++;
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
