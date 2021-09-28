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

#include <stdio.h>
#include <stdlib.h>

#include "common/xf_headers.hpp"
#include "xf_sobel_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }
    char in[100], in1[100], out_hlsx[100], out_ocvx[100];
    char out_errorx[100], out_hlsy[100], out_ocvy[100], out_errory[100];

    cv::Mat in_img, in_gray, diff;
    cv::Mat c_grad_x_1, c_grad_y_1;
    cv::Mat c_grad_x, c_grad_y;
    cv::Mat hls_grad_x, hls_grad_y;
    cv::Mat diff_grad_x, diff_grad_y;

// reading in the color image
#if GRAY
    in_img = cv::imread(argv[1], 0);
#else
    in_img = cv::imread(argv[1], 1);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", in);
        return 0;
    }

    ///////////////// 	Opencv  Reference  ////////////////////////
    int scale = 1;
    int delta = 0;

#if T_8U

#if (FILTER_WIDTH != 7)

    int ddepth = CV_8U;
    typedef unsigned char TYPE; // Should be short int when ddepth is CV_16S
#if GRAY
#define PTYPE CV_8UC1 // Should be CV_16S when ddepth is CV_16S
#else
#define PTYPE CV_8UC3 // Should be CV_16S when ddepth is CV_16S
#endif
#endif
#if (FILTER_WIDTH == 7)

    int ddepth = -1;            // CV_32F;	//Should be CV_32F if the output pixel type is XF_32UC1
    typedef unsigned char TYPE; // Should be int when ddepth is CV_32F
#if GRAY
#define PTYPE CV_8UC1 // Should be CV_16S when ddepth is CV_16S
#else
#define PTYPE CV_8UC3 // Should be CV_16S when ddepth is CV_16S
#endif

#endif

#else

#if (FILTER_WIDTH != 7)

    int ddepth = CV_16S;
    typedef short TYPE; // Should be short int when ddepth is CV_16S
#if GRAY
#define PTYPE CV_16SC1 // Should be CV_16S when ddepth is CV_16S
#else
#define PTYPE CV_16SC3 // Should be CV_16S when ddepth is CV_16S
#endif
#endif
#if (FILTER_WIDTH == 7)

    int ddepth = CV_16S; // CV_32F;	//Should be CV_32F if the output pixel type is XF_32UC1
    typedef short TYPE;  // Should be int when ddepth is CV_32F
#if GRAY
#define PTYPE CV_16SC1 // Should be CV_16S when ddepth is CV_16S
#else
#define PTYPE CV_16SC3 // Should be CV_16S when ddepth is CV_16S
#endif

#endif

#endif

    // create memory for output images
    hls_grad_x.create(in_img.rows, in_img.cols, PTYPE);
    hls_grad_y.create(in_img.rows, in_img.cols, PTYPE);
    diff_grad_x.create(in_img.rows, in_img.cols, PTYPE);
    diff_grad_y.create(in_img.rows, in_img.cols, PTYPE);
    c_grad_x_1.create(in_img.rows, in_img.cols, PTYPE);
    c_grad_y_1.create(in_img.rows, in_img.cols, PTYPE);

    cv::Sobel(in_img, c_grad_x_1, ddepth, 1, 0, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);
    cv::Sobel(in_img, c_grad_y_1, ddepth, 0, 1, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);

    imwrite("out_ocvx.jpg", c_grad_x_1);
    imwrite("out_ocvy.jpg", c_grad_y_1);

    unsigned short rows = in_img.rows;
    unsigned short cols = in_img.cols;

    // Call the top function
    sobel_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)hls_grad_x.data,
                (ap_uint<OUTPUT_PTR_WIDTH>*)hls_grad_y.data, rows, cols);

//////////////////  Compute Absolute Difference ////////////////////
#if (FILTER_WIDTH == 3 | FILTER_WIDTH == 5)
    absdiff(c_grad_x_1, hls_grad_x, diff_grad_x);
    absdiff(c_grad_y_1, hls_grad_y, diff_grad_y);
#endif
#if (FILTER_WIDTH == 7)
    if (OUT_TYPE == XF_8UC1 || OUT_TYPE == XF_16SC1 || OUT_TYPE == XF_8UC3 || OUT_TYPE == XF_16SC3) {
        absdiff(c_grad_x_1, hls_grad_x, diff_grad_x);
        absdiff(c_grad_y_1, hls_grad_y, diff_grad_y);
    } else if (OUT_TYPE == XF_32UC1) {
        c_grad_x_1.convertTo(c_grad_x, CV_32S);
        c_grad_y_1.convertTo(c_grad_y, CV_32S);
        absdiff(c_grad_x, hls_grad_x, diff_grad_x);
        absdiff(c_grad_y, hls_grad_y, diff_grad_y);
    }
#endif

    imwrite("out_errorx.jpg", diff_grad_x);
    imwrite("out_errory.jpg", diff_grad_y);

    imwrite("hlsx.jpg", hls_grad_x);
    imwrite("hlsy.jpg", hls_grad_y);

    float err_per, err_per1;
    int ret;

    xf::cv::analyzeDiff(diff_grad_x, 0, err_per);
    xf::cv::analyzeDiff(diff_grad_y, 0, err_per1);

    if (err_per > 0.0f) {
        fprintf(stderr, "Test failed .... !!!\n ");
        ret = 1;
    } else {
        std::cout << "Test Passed .... !!!" << std::endl;
        ret = 0;
    }

    return ret;
}
