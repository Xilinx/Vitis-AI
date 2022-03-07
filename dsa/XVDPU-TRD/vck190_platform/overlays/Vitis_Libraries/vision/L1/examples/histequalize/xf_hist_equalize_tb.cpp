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
#include "xf_hist_equalize_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_img_copy, out_img, ocv_ref, diff;

    // reading in the color image
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }

    int height = in_img.rows;
    int width = in_img.cols;

    // create memory for output images
    in_img.copyTo(in_img_copy);
    out_img.create(height, width, XF_8UC1);
    ocv_ref.create(height, width, XF_8UC1);
    diff.create(height, width, XF_8UC1);

    ///////////////// 	Opencv  Reference  ////////////////////////
    cv::equalizeHist(in_img, ocv_ref);

    ///////////////// Call the top function ///////////////////////

    equalizeHist_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, (ap_uint<INPUT_PTR_WIDTH>*)in_img_copy.data,
                       (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, height, width);

    //////////////////  Compute Absolute Difference ////////////////////
    cv::absdiff(ocv_ref, out_img, diff);

    cv::imwrite("input.jpg", in_img);
    cv::imwrite("out_ocv.jpg", ocv_ref);
    cv::imwrite("out_hls.jpg", out_img);
    cv::imwrite("out_error.jpg", diff);

    // Find minimum and maximum differences.
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    return 0;
}
