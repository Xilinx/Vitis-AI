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

#ifndef _XF_SW_UTILS_H_
#define _XF_SW_UTILS_H_

#include "xf_common.hpp"
#include <iostream>
#include <cmath>

namespace xf {
namespace cv {

template <int _PTYPE, int _ROWS, int _COLS, int _NPC>
xf::cv::Mat<_PTYPE, _ROWS, _COLS, _NPC> imread(char* img, int type) {
    ::cv::Mat img_load = ::cv::imread(img, type);

    xf::cv::Mat<_PTYPE, _ROWS, _COLS, _NPC> input(img_load.rows, img_load.cols);

    if (img_load.data == NULL) {
        fprintf(stderr, "\nError : Couldn't open the image at %s\n ", img);
        exit(-1);
    }

    input.copyTo(img_load.data);

    return input;
}

template <int _PTYPE, int _ROWS, int _COLS, int _NPC>
void imwrite(const char* str, xf::cv::Mat<_PTYPE, _ROWS, _COLS, _NPC>& output) {
    int list_ptype[] = {CV_8UC1,  CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_32SC1,
                        CV_16UC1, CV_32SC1, CV_8UC1,  CV_8UC3,  CV_16UC3, CV_16SC3};
    int _PTYPE_CV = list_ptype[_PTYPE];

    ::cv::Mat input(output.rows, output.cols, _PTYPE_CV);
    input.data = output.copyFrom();
    ::cv::imwrite(str, input);
}

template <int _PTYPE, int _ROWS, int _COLS, int _NPC>
void absDiff(::cv::Mat& cv_img, xf::cv::Mat<_PTYPE, _ROWS, _COLS, _NPC>& xf_img, ::cv::Mat& diff_img) {
    assert((cv_img.rows == xf_img.rows) && (cv_img.cols == xf_img.cols) && "Sizes of cv and xf images should be same");
    assert((xf_img.rows == diff_img.rows) && (xf_img.cols == diff_img.cols) &&
           "Sizes of xf and diff images should be same");
    assert(((_NPC == XF_NPPC8) || (_NPC == XF_NPPC4) || (_NPC == XF_NPPC2) || (_NPC == XF_NPPC1)) &&
           "Only XF_NPPC1, XF_NPPC2, XF_NPPC4, XF_NPPC8 are supported");
    assert((cv_img.channels() == XF_CHANNELS(_PTYPE, _NPC)) && "Number of channels of cv and xf images does not match");

    int cv_bitdepth = 8;
    int num_chnls = cv_img.channels();
    int cv_nbytes = 1;

    if (cv_img.depth() == CV_8U) {
        cv_bitdepth = 8;
    } else if (cv_img.depth() == CV_16S || cv_img.depth() == CV_16U) {
        cv_bitdepth = 16;
    } else if (cv_img.depth() == CV_32S || cv_img.depth() == CV_32F) {
        cv_bitdepth = 32;
    } else {
        fprintf(stderr, "OpenCV image's depth not supported\n ");
        return;
    }

    int cv_pixdepth = cv_bitdepth * num_chnls;
    cv_nbytes = cv_bitdepth / 8;

    int ch = 0;
    int xf_npc_idx = 0;
    int diff_ptr = 0;
    int xf_ptr = 0;
    int cv_ptr = 0;
    XF_CTUNAME(_PTYPE, _NPC) cv_val = 0, xf_val = 0, diff_val = 0;
    XF_TNAME(_PTYPE, _NPC) xf_val_total = 0;

    for (int r = 0, xf_ptr = 0; r < cv_img.rows; ++r) {
        for (int c = 0; c<cv_img.cols>> XF_BITSHIFT(_NPC); ++c) {
#ifdef __SDSVHLS__
            xf_val_total = xf_img.data[xf_ptr++];

            for (int b = 0; b < _NPC; ++b) {
                for (int c = 0; c < num_chnls; ++c) {
                    for (int l = 0; l < cv_nbytes; ++l, ++xf_npc_idx) {
                        cv_val.range(((l + 1) * 8) - 1, l * 8) = cv_img.data[cv_ptr++];
                        xf_val.range(((l + 1) * 8) - 1, l * 8) =
                            xf_val_total.range(((xf_npc_idx + 1) * 8) - 1, xf_npc_idx * 8);
                    }
                    diff_val = __ABS((int)cv_val - (int)xf_val);
                    for (int l = 0; l < cv_nbytes; ++l) {
                        diff_img.data[diff_ptr++] = diff_val.range(((l + 1) * 8) - 1, l * 8);
                    }
                }
            }
            xf_npc_idx = 0;

#else
            for (int xf_npc_idx = 0; xf_npc_idx < _NPC; ++xf_npc_idx) {
                for (int ch = 0; ch < num_chnls; ++ch) {
                    xf_val = xf_img.data[xf_ptr].chnl[xf_npc_idx][ch];

                    for (int b = 0; b < cv_nbytes; ++b) {
                        cv_val.range(((b + 1) * 8) - 1, b * 8) = cv_img.data[cv_ptr++];
                    }
                    diff_val = __ABS((int)cv_val - (int)xf_val);
                    for (int b = 0; b < cv_nbytes; ++b) {
                        diff_img.data[diff_ptr++] = diff_val.range(((b + 1) * 8) - 1, b * 8);
                    }
                }
            }
            ++xf_ptr;
#endif
        }
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
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;
}
} // namespace cv
} // namespace xf

#endif
