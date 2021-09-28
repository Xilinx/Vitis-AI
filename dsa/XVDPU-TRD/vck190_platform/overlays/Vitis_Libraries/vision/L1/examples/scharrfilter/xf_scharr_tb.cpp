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
#include "xf_scharr_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_gray;
    cv::Mat c_grad_x, c_grad_y;
    cv::Mat hls_grad_x, hls_grad_y;
    cv::Mat diff_grad_x, diff_grad_y;

// reading in the gray image
#if GRAY
    in_img = cv::imread(argv[1], 0);
#else
    in_img = cv::imread(argv[1], 1);
#endif

#if T_8U

    int ddepth = CV_8U;
#if GRAY
#define PTYPE CV_8UC1 // Should be CV_16S when ddepth is CV_16S
#else
#define PTYPE CV_8UC3 // Should be CV_16S when ddepth is CV_16S
#endif

    typedef unsigned char TYPE; // short int TYPE; //
#else

    int ddepth = CV_16S;
#if GRAY
#define PTYPE CV_16SC1 // Should be CV_16S when ddepth is CV_16S
#else
#define PTYPE CV_16SC3 // Should be CV_16S when ddepth is CV_16S
#endif
    typedef unsigned short TYPE; // short int TYPE; //
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }

    // create memory for output images
    c_grad_x.create(in_img.rows, in_img.cols, PTYPE);
    c_grad_y.create(in_img.rows, in_img.cols, PTYPE);
    hls_grad_x.create(in_img.rows, in_img.cols, PTYPE);
    hls_grad_y.create(in_img.rows, in_img.cols, PTYPE);
    diff_grad_x.create(in_img.rows, in_img.cols, PTYPE);
    diff_grad_y.create(in_img.rows, in_img.cols, PTYPE);

    ////////////    Opencv Reference    //////////////////////
    int scale = 1;
    int delta = 0;

    Scharr(in_img, c_grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_CONSTANT);
    Scharr(in_img, c_grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_CONSTANT);

    imwrite("out_ocvx.jpg", c_grad_x);
    imwrite("out_ocvy.jpg", c_grad_y);

    int rows = in_img.rows;
    int cols = in_img.cols;

    // Call the top function

    scharr_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)hls_grad_x.data,
                 (ap_uint<OUTPUT_PTR_WIDTH>*)hls_grad_y.data, rows, cols);

    imwrite("out_hlsx.jpg", hls_grad_x);
    imwrite("out_hlsy.jpg", hls_grad_y);

    //////////////////  Compute Absolute Difference ////////////////////

    absdiff(c_grad_x, hls_grad_x, diff_grad_x);
    absdiff(c_grad_y, hls_grad_y, diff_grad_y);

    imwrite("out_errorx.jpg", diff_grad_x);
    imwrite("out_errory.jpg", diff_grad_y);

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    double minval1 = 256, maxval1 = 0;
    int cnt = 0, cnt1 = 0;
    float err_per, err_per1;
    xf::cv::analyzeDiff(diff_grad_x, 0, err_per);
    xf::cv::analyzeDiff(diff_grad_y, 0, err_per1);

    int ret = 0;
    if (err_per > 0.0f) {
        fprintf(stderr, "Test failed .... !!!\n ");
        ret = 1;
    } else {
        std::cout << "Test Passed .... !!!" << std::endl;
        ret = 0;
    }

    return ret;
}
