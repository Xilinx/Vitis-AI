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

#ifndef _XF_AIE_SW_UTILS_H_
#define _XF_AIE_SW_UTILS_H_

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef PROFILE
std::chrono::time_point<std::chrono::high_resolution_clock> start;
std::chrono::time_point<std::chrono::high_resolution_clock> stop;
std::chrono::microseconds tdiff;
#define START_TIMER start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)                                                                                       \
    stop = std::chrono::high_resolution_clock::now();                                                          \
    tdiff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);                               \
    std::cout << "RUNTIME of " << name << ": " << ((float)tdiff.count() / (float)1000) << " ms " << std::endl; \
    std::cout << "FPS of " << name << ": " << (1000000 / tdiff.count()) << " fps " << std::endl;
#else
#define START_TIMER
#define STOP_TIMER(name)
#endif

extern "C" {
void printConvolutionWindow(cv::Mat& img, int row, int col, int kernel_size_rows, int kernel_size_cols) {
    for (int m = -(kernel_size_rows >> 1); m <= (kernel_size_rows >> 1); m++) {
        for (int n = -(kernel_size_cols >> 1); n <= (kernel_size_cols >> 1); n++) {
            int y = std::min(std::max((row + m), 0), (img.rows - 1));
            int x = std::min(std::max((col + n), 0), (img.cols - 1));
            ;
            int val = (int)img.at<unsigned char>(y, x);
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void analyzeDiff(::cv::Mat& diff_img, int err_thresh, float& err_per) {
    int cv_bitdepth;
    if (diff_img.depth() == CV_8U) {
        cv_bitdepth = 8;
    } else if (diff_img.depth() == CV_16U || diff_img.depth() == CV_16S) {
        cv_bitdepth = 16;
    } else if (diff_img.depth() == CV_32S || diff_img.depth() == CV_32F) {
        cv_bitdepth = 32;
    } else {
        fprintf(stderr, "OpenCV image's depth not supported for this function\n ");
        return;
    }

    int cnt = 0;
    double minval = std::pow(2.0, cv_bitdepth), maxval = 0;
    int max_fix = (int)(std::pow(2.0, cv_bitdepth) - 1.0);
    for (int i = 0; i < diff_img.rows; i++) {
        for (int j = 0; j < diff_img.cols; j++) {
            int v = 0;
            for (int k = 0; k < diff_img.channels(); k++) {
                int v_tmp = 0;
                if (diff_img.channels() == 1) {
                    if (cv_bitdepth == 8)
                        v_tmp = (int)diff_img.at<unsigned char>(i, j);
                    else if (cv_bitdepth == 16 && diff_img.depth() == CV_16U) // 16 bitdepth
                        v_tmp = (int)diff_img.at<unsigned short>(i, j);
                    else if (cv_bitdepth == 16 && diff_img.depth() == CV_16S) // 16 bitdepth
                        v_tmp = (int)diff_img.at<short>(i, j);
                    else if (cv_bitdepth == 32 && diff_img.depth() == CV_32S)
                        v_tmp = (int)diff_img.at<int>(i, j);
                } else // 3 channels
                    v_tmp = (int)diff_img.at< ::cv::Vec3b>(i, j)[k];

                if (v_tmp > v) v = v_tmp;
            }
            if (v > err_thresh) {
                cnt++;
                if (diff_img.depth() == CV_8U)
                    diff_img.at<unsigned char>(i, j) = max_fix;
                else if (diff_img.depth() == CV_16U)
                    diff_img.at<unsigned short>(i, j) = max_fix;
                else if (diff_img.depth() == CV_16S)
                    diff_img.at<short>(i, j) = max_fix;
                else if (diff_img.depth() == CV_32S)
                    diff_img.at<int>(i, j) = max_fix;
                else
                    diff_img.at<float>(i, j) = (float)max_fix;
            }
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    err_per = 100.0 * (float)cnt / (diff_img.rows * diff_img.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tNumber of pixels above error threshold = " << cnt << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;
}
}

#endif
