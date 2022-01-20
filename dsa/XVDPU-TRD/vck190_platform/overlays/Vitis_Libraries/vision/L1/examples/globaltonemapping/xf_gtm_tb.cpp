/*
 * Copyright 2021 Xilinx, Inc.
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
#include <iostream>
#include <math.h>

#include "xf_gtm_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <INPUT IMAGE PATH > \n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat hdr_img, in_xyz, out_img, out_hls, matlab_y, diff;
    cv::Mat xyzchannel[3], _xyzchannel[3];

    // Reading in the images:
    hdr_img = cv::imread(argv[1], -1);

    if (hdr_img.data == NULL) {
        printf("ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    diff.create(hdr_img.rows, hdr_img.cols, CV_8UC1);

    in_xyz.create(hdr_img.rows, hdr_img.cols, CV_16UC3);
    out_img.create(hdr_img.rows, hdr_img.cols, CV_8UC3);
    out_hls.create(hdr_img.rows, hdr_img.cols, CV_8UC3);

    int height = hdr_img.rows;
    int width = hdr_img.cols;

    cv::cvtColor(hdr_img, in_xyz, cv::COLOR_BGR2XYZ);
    cv::split(in_xyz, xyzchannel);

    _xyzchannel[0].create(hdr_img.rows, hdr_img.cols, CV_8UC1);
    _xyzchannel[1].create(hdr_img.rows, hdr_img.cols, CV_8UC1);
    _xyzchannel[2].create(hdr_img.rows, hdr_img.cols, CV_8UC1);

    float c1 = 3.0;
    float c2 = 1.5;

    double maxL = 0, minL = 100;
    double mean = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float pxl_val = log10(xyzchannel[1].at<ushort>(i, j));

            mean = mean + pxl_val;
            maxL = (maxL > pxl_val) ? maxL : pxl_val;
            minL = (minL < pxl_val) ? minL : pxl_val;
        }
    }
    mean = mean / (height * width);

    float maxLd, minLd;
    maxLd = 2.4;
    minLd = 0;

    float K1 = (maxLd - minLd) / (maxL - minL);

    float d0 = maxL - minL;
    float sigma_sq = (c1 * c1) / (2 * d0 * d0);
    float val, out_val;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            val = log10(xyzchannel[1].at<ushort>(i, j));
            val = val - mean;

            float K2 = (1 - K1) * exp(-(val * val * sigma_sq)) + K1;
            out_val = exp(c2 * K2 * val + mean);

            int x_val = xyzchannel[0].at<ushort>(i, j);
            int y_val = xyzchannel[1].at<ushort>(i, j);
            int z_val = xyzchannel[2].at<ushort>(i, j);

            _xyzchannel[0].at<uchar>(i, j) = (uint8_t)((out_val / y_val) * x_val);
            _xyzchannel[2].at<uchar>(i, j) = (uint8_t)((out_val / y_val) * z_val);
            _xyzchannel[1].at<uchar>(i, j) = (uint8_t)out_val;
        }
    }
    cv::Mat out_xyz;
    cv::merge(_xyzchannel, 3, out_xyz);
    cv::cvtColor(out_xyz, out_img, cv::COLOR_XYZ2BGR);

    ////////////Top function call //////////////////
    for (int i = 0; i < 2; i++) {
        // Call the top function
        gtm_accel((ap_uint<INPUT_PTR_WIDTH>*)hdr_img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_hls.data, c1, c2, height,
                  width);
    }

    imwrite("out_img.jpg", out_img);
    imwrite("out_hls.jpg", out_hls);

    // Compute absolute difference image
    cv::absdiff(out_img, out_hls, diff);

    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    return 0;
}