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
#include "xf_modefilter_config.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;
cv::RNG rng(12345);
void mode_filter_rgb(cv::Mat _src, cv::Mat _dst, int win_sz) {
    int win_sz_sq = win_sz * win_sz;
    int window[win_sz_sq];
    cv::Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    cv::Mat _src_border;

    _src_border.create(_src.rows + win_sz - 1, _src.cols + win_sz - 1, CV_8UC3);

    int border = floor(win_sz / 2);

    cv::copyMakeBorder(_src, _src_border, border, border, border, border, cv::BORDER_REPLICATE, value);

    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < _src.rows; i++) {
            for (int j = 0; j < _src.cols; j++) {
                for (int p = 0; p < win_sz; p++) {
                    for (int q = 0; q < win_sz; q++) {
                        // cout<<p<<" "<<q<<" "<<endl;
                        window[q + p * win_sz] = _src_border.at<cv::Vec3b>(i + p, j + q)[k];
                    }
                }

                int max_count = 0, idx = 0;

                for (int m = 0; m < win_sz_sq; m++) {
                    int count = 1;
                    for (int n = m + 1; n < win_sz_sq - 1; n++) {
                        if (window[m] == window[n]) count++;
                    }
                    if (count > max_count) {
                        max_count = count;
                    }
                }

                for (int m = 0; m < win_sz_sq; m++) {
                    int count = 1;
                    for (int n = m + 1; n < win_sz_sq - 1; n++) {
                        if (window[m] == window[n]) count++;
                    }

                    if (count == max_count) {
                        idx = m;
                    }
                }

                _dst.at<cv::Vec3b>(i, j)[k] = window[idx];
            }
        }
    }
    return;
}
void mode_filter_gray(cv::Mat _src, cv::Mat _dst, int win_sz) {
    int win_sz_sq = win_sz * win_sz;
    int window[win_sz_sq];
    int i_1_index = 0, j_1_index = 0, i_plus_index = 0, j_plus_index = 0;

    cv::Mat _src_border;

    _src_border.create(_src.rows + win_sz - 1, _src.cols + win_sz - 1, CV_8UC1);

    int border = floor(win_sz / 2);

    cv::copyMakeBorder(_src, _src_border, border, border, border, border, cv::BORDER_REPLICATE);

    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            for (int p = 0; p < win_sz; p++) {
                for (int q = 0; q < win_sz; q++) {
                    window[q + p * win_sz] = _src_border.at<uchar>(i + p, j + q);
                }
            }
            int max_count = 0, idx = 0;

            for (int i = 0; i < win_sz_sq; i++) {
                int count = 1;
                for (int j = i + 1; j < win_sz_sq - 1; j++) {
                    if (window[i] == window[j]) count++;
                }
                if (count > max_count) {
                    max_count = count;
                }
            }

            for (int i = 0; i < win_sz_sq; i++) {
                int count = 1;
                for (int j = i + 1; j < win_sz_sq - 1; j++) {
                    if (window[i] == window[j]) count++;
                }

                if (count == max_count) {
                    idx = i;
                }
            }

            _dst.at<uchar>(i, j) = window[idx];
        }
    }

    return;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image path>\n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_ref, ocv_ref1, diff;

//  Reading in the image:
#if GRAY
    cout << "gray:";
    in_img = cv::imread(argv[1], 0); // reading in the gray image
    imwrite("in_img1.jpg", in_img);
#else
    in_img = cv::imread(argv[1], 1); // reading in the color image
    imwrite("in_img2.jpg", in_img);
#endif

    if (!in_img.data) {
        return -1;
    }

// imwrite("in_img.jpg", in_img);

// create memory for output image
#if GRAY
    cout << "gray:";
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1);
    ocv_ref1.create(in_img.rows, in_img.cols, CV_8UC1);
    out_img.create(in_img.rows, in_img.cols, CV_8UC1); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC1);
#else
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3);
    ocv_ref1.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img.create(in_img.rows, in_img.cols, CV_8UC3); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
#endif

#if GRAY
    cout << "gray:";
    mode_filter_gray(in_img, ocv_ref, WINDOW_SIZE);
#else
    cout << "rgb";
    mode_filter_rgb(in_img, ocv_ref, WINDOW_SIZE);
#endif

    modefilter_accel((ap_uint<PTR_WIDTH>*)in_img.data, in_img.rows, in_img.cols, (ap_uint<PTR_WIDTH>*)out_img.data);

    imwrite("out_img.jpg", out_img);
    imwrite("ocv_ref.jpg", ocv_ref);
    // imwrite("ocv_ref.jpg", ocv_ref1);

    cv::absdiff(ocv_ref, out_img, diff);

    imwrite("diff.jpg", diff);
    absdiff(ocv_ref, out_img, diff);
    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 0, err_per);

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return 1;
    }
    std::cout << "Test Passed " << std::endl;

    return 0;
}
