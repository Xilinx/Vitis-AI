/*
 * Copyright 2020 Xilinx, Inc.
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
#include "xf_lensshading_config.h"
#include <math.h>

// OpenCV reference function:
void LSC_ref(cv::Mat& _src, cv::Mat& _dst) {
    int center_pixel_pos_x = (_src.cols / 2);
    int center_pixel_pos_y = (_src.rows / 2);
    float max_distance = std::sqrt((_src.rows - center_pixel_pos_y) * (_src.rows - center_pixel_pos_y) +
                                   (_src.cols - center_pixel_pos_x) * (_src.cols - center_pixel_pos_x));

    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                float distance = std::sqrt((center_pixel_pos_y - i) * (center_pixel_pos_y - i) +
                                           (center_pixel_pos_x - j) * (center_pixel_pos_x - j)) /
                                 max_distance;

                float gain = (0.01759 * ((distance + 28.37) * (distance + 28.37))) - 13.36;
#if T_8U
                int value = (_src.at<cv::Vec3b>(i, j)[k] * gain);
                if (value > 255) {
                    value = 255;
                }
                _dst.at<cv::Vec3b>(i, j)[k] = (unsigned char)value;
#else
                int value = (_src.at<cv::Vec3w>(i, j)[k] * gain);
                if (value > 65535) {
                    value = 65535;
                }
                _dst.at<cv::Vec3w>(i, j)[k] = (unsigned short)value;

#endif
            }
        }
    }
}

int main(int argc, char** argv) {
    cv::Mat in_img, out_img, out_img_hls, diff;

#if T_8U
    in_img = cv::imread(argv[1], 1); // read image
#else
    in_img = cv::imread(argv[1], -1); // read image
#endif
    if (!in_img.data) {
        return -1;
    }

    imwrite("in_img.png", in_img);

#if T_8U
    out_img.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img_hls.create(in_img.rows, in_img.cols, CV_8UC3);
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
#else
    out_img.create(in_img.rows, in_img.cols, CV_16UC3);
    out_img_hls.create(in_img.rows, in_img.cols, CV_16UC3);
    diff.create(in_img.rows, in_img.cols, CV_16UC3);
#endif

    LSC_ref(in_img, out_img);

    int height = in_img.rows;
    int width = in_img.cols;

    lensshading_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img_hls.data, height,
                      width);

    // Write output image
    cv::imwrite("hls_out.jpg", out_img_hls);
    cv::imwrite("ocv_out.jpg", out_img);

    // Compute absolute difference image
    cv::absdiff(out_img_hls, out_img, diff);
    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    if (err_per > 0.0f) {
        return 1;
    }
    return 0;
}
