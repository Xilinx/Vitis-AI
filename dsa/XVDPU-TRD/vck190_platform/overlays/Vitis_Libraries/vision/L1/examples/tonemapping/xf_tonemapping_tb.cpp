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
#include "xf_tonemapping_config.h"
int main(int argc, char** argv) {
    cv::Mat in_img;

    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image>\n");
        return -1;
    }

    // Read input image
    in_img = cv::imread(argv[1], 1);
    if (in_img.data == NULL) {
        fprintf(stderr, "Can't open image !!\n");
        return -1;
    }

    int cols = in_img.cols;
    int rows = in_img.rows;
    int blk_height = 128;
    int blk_width = 128;

    std::cout << "Block height : " << blk_height << std::endl;
    std::cout << "Block width  : " << blk_width << std::endl;

    cv::Mat out_img_tmp(rows, cols, CV_8UC3);
    cv::Mat out_img(rows, cols, CV_8UC3);
    ////////////Top function call //////////////////
    tonemapping_accel((ap_uint<IN_PTR_WIDTH>*)in_img.data, (ap_uint<OUT_PTR_WIDTH>*)out_img_tmp.data, rows, cols,
                      blk_height, blk_width);
    tonemapping_accel((ap_uint<IN_PTR_WIDTH>*)in_img.data, (ap_uint<OUT_PTR_WIDTH>*)out_img.data, rows, cols,
                      blk_height, blk_width);

    // Write output image
    imwrite("hls_out.png", out_img);

    std::cout << "Testbench done" << std::endl;

    return 0;
}
