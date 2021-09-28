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
#include "xf_opencl_wrap.hpp"
int main(int argc, char** argv) {
    cv::Mat in_img;

    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image>\n");
        return -1;
    }

    // Read input image
    in_img = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (in_img.data == NULL) {
        fprintf(stderr, "Can't open image !!\n");
        return -1;
    }

    int rows = in_img.rows;
    int cols = in_img.cols;
    int blk_height = 0;
    int blk_width = 0;

    for (int i = 0; i < 32; i++) {
        if ((1 << i) > (rows / 2)) break;
        blk_height = (1 << i);
    }

    for (int i = 0; i < 32; i++) {
        if ((1 << i) > (cols / 2)) break;
        blk_width = (1 << i);
    }

    std::cout << "Block height : " << blk_height << std::endl;
    std::cout << "Block width  : " << blk_width << std::endl;

    cv::Mat out_img_tmp(rows, cols, CV_8UC3);
    cv::Mat out_img(rows, cols, CV_8UC3);
    ////////////Top function call //////////////////
    cl_kernel_wrapper* krnl1 =
        cl_kernel_mgr::registerKernel("tonemapping_accel", "krnl_tonemapping", XCLIN(in_img), XCLOUT(out_img_tmp),
                                      XCLIN(rows), XCLIN(cols), XCLIN(blk_height), XCLIN(blk_width));

    cl_kernel_wrapper* krnl2 =
        cl_kernel_mgr::registerKernel("tonemapping_accel", "krnl_tonemapping", XCLIN(in_img), XCLOUT(out_img),
                                      XCLIN(rows), XCLIN(cols), XCLIN(blk_height), XCLIN(blk_width));
    krnl2->addDependent(krnl1);
    cl_kernel_mgr::exec_all();

    // Write output image
    cv::imwrite("hls_out.png", out_img);

    std::cout << "Testbench done" << std::endl;

    return 0;
}
