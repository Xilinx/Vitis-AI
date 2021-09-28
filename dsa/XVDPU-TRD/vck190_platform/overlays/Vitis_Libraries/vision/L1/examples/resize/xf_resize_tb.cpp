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
#include "xf_resize_config.h"

int main(int argc, char** argv) {
    cv::Mat img, out_img, result_ocv, error;

    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image>\n");
        return -1;
    }

#if GRAY
    img.create(cv::Size(WIDTH, HEIGHT), CV_8UC1);
    out_img.create(cv::Size(NEWWIDTH, NEWHEIGHT), CV_8UC1);
    result_ocv.create(cv::Size(NEWWIDTH, NEWHEIGHT), CV_8UC1);
    error.create(cv::Size(NEWWIDTH, NEWHEIGHT), CV_8UC1);
#else
    img.create(cv::Size(WIDTH, HEIGHT), CV_8UC3);
    result_ocv.create(cv::Size(NEWWIDTH, NEWHEIGHT), CV_8UC3);
    out_img.create(cv::Size(NEWWIDTH, NEWHEIGHT), CV_8UC3);
    error.create(cv::Size(NEWWIDTH, NEWHEIGHT), CV_8UC3);
#endif

#if GRAY
    // reading in the color image
    img = cv::imread(argv[1], 0);
#else
    img = cv::imread(argv[1], 1);
#endif

    if (!img.data) {
        return -1;
    }

    cv::imwrite("input.png", img);

    unsigned short in_width, in_height;
    unsigned short out_width, out_height;

    in_width = img.cols;
    in_height = img.rows;
    out_height = NEWHEIGHT;
    out_width = NEWWIDTH;

/*OpenCV resize function*/

#if INTERPOLATION == 0
    cv::resize(img, result_ocv, cv::Size(out_width, out_height), 0, 0, cv::INTER_NEAREST);
#endif
#if INTERPOLATION == 1
    cv::resize(img, result_ocv, cv::Size(out_width, out_height), 0, 0, cv::INTER_LINEAR);
#endif
#if INTERPOLATION == 2
    cv::resize(img, result_ocv, cv::Size(out_width, out_height), 0, 0, cv::INTER_AREA);
#endif

    /* Call the top function */
    resize_accel((ap_uint<INPUT_PTR_WIDTH>*)img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, in_height, in_width,
                 out_height, out_width);

    float err_per;
    cv::absdiff(result_ocv, out_img, error);
    xf::cv::analyzeDiff(error, 5, err_per);
    cv::imwrite("hls_out.png", out_img);
    cv::imwrite("resize_ocv.png", result_ocv);
    cv::imwrite("error.png", error);

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    }

    return 0;
}
