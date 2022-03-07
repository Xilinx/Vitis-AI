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
#include "xf_channel_extract_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <Input image>\n");
        return -1;
    }

    cv::Mat in_src, in_rgba, out_img;
    // read image
    in_src = cv::imread(argv[1], 1);

    if (in_src.data == NULL) {
        fprintf(stderr, "Cannot open image \n");
        return 0;
    }

    out_img.create(in_src.rows, in_src.cols, CV_8U);
    cv::cvtColor(in_src, in_rgba, cv::COLOR_BGR2RGBA);
    uint16_t channel = XF_EXTRACT_CH_R;

    int height = in_rgba.rows;
    int width = in_rgba.cols;

    // Call the top function
    channel_extract_accel((ap_uint<INPUT_PTR_WIDTH>*)in_rgba.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, channel,
                          height, width);

    cv::imwrite("hls_out.png", out_img);

    std::vector<cv::Mat> bgr_planes;
    // call OpenCV function
    cv::split(in_src, bgr_planes);
    // write output and OpenCV reference image
    cv::imwrite("out_ocv.png", bgr_planes[2]);

    cv::Mat diff;
    diff.create(in_src.rows, in_src.cols, CV_8U);

    // Check with the correct channel. Keep 2 for R, 1 for G and 0 for B in index of bgr_planes
    cv::absdiff(bgr_planes[2], out_img, diff);
    cv::imwrite("diff.jpg", diff);

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            unsigned char v = diff.at<unsigned char>(i, j);
            if (v > 0) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    float err_per = 100.0 * (float)cnt / (in_src.rows * in_src.cols);

    std::cout << "Minimum error in intensity =" << minval << "\t"
              << "Maximum error in intensity = " << maxval << "\t"
              << "Percentage of pixels above error" << err_per << std::endl;

    if (err_per > 0.0f) return -1;

    return 0;
}
