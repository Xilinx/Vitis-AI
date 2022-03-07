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
#include "xf_axiconv_config.h"
#include "common/xf_axi.hpp"

using namespace std;

#define _W 8

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage: <executable> <image>\n");
        return -1;
    }

    cv::Mat img, diff;
    img = cv::imread(argv[1], 0);
    if (img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    int rows = img.rows;
    int cols = img.cols;
    cv::Mat out_img(rows, cols, CV_8UC1);

    // convert input to axiStream
    hls::stream<ap_axiu<_W, 1, 1, 1> > _src;
    xf::cv::cvMat2AXIvideoxf<XF_NPPC1, _W>(img, _src);

    // output axiStream
    hls::stream<ap_axiu<_W, 1, 1, 1> > _dst;

    // Launch the kernel
    axiconv_accel(_src, _dst, rows, cols);

    xf::cv::AXIvideo2cvMatxf<XF_NPPC1>(_dst, out_img);

    // Write output image
    cv::imwrite("output.png", out_img);

    /**** validation ****/
    // diff
    diff.create(img.rows, img.cols, CV_8UC1);
    // Compute absolute difference image
    cv::absdiff(img, out_img, diff);
    imwrite("error.png", diff); // Save the difference image for debugging purpose
    float err_per;
    xf::cv::analyzeDiff(diff, 0, err_per);
    if (err_per > 0.0f) {
        return 1;
    }
    /**** end of validation ****/

    return 0;
}
