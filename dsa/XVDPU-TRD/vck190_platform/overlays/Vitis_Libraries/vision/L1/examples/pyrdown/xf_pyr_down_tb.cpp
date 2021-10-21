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
#include "xf_pyr_down_config.h"

int main(int argc, char* argv[]) {
    cv::Mat input_image, output_image, output_diff_xf_cv, output_xf;

#if RGBA
    input_image = cv::imread(argv[1], 1);
#else
    input_image = cv::imread(argv[1], 0);
#endif

    int input_height = input_image.rows;
    int input_width = input_image.cols;

    int output_height = (input_image.rows + 1) >> 1;
    int output_width = (input_image.cols + 1) >> 1;

#if RGBA
    output_xf.create(output_height, output_width, CV_8UC3);
    output_diff_xf_cv.create(output_height, output_width, CV_8UC3);
#else
    output_xf.create(output_height, output_width, CV_8UC1);
    output_diff_xf_cv.create(output_height, output_width, CV_8UC1);
#endif

    std::cout << "Input Height " << input_height << " Input_Width " << input_width << std::endl;
    std::cout << "Output Height " << output_height << " Output_Width " << output_width << std::endl;

    cv::pyrDown(input_image, output_image, cv::Size(output_width, output_height), cv::BORDER_REPLICATE);

    cv::imwrite("opencv_image.png", output_image);
    ///////////////   End of OpenCV reference     /////////////////

    ////////////////////	HLS TOP function call	/////////////////

    pyr_down_accel((ap_uint<INPUT_PTR_WIDTH>*)input_image.data, (ap_uint<OUTPUT_PTR_WIDTH>*)output_xf.data,
                   input_height, input_width, output_height, output_width);

    float err_per;
    cv::absdiff(output_image, output_xf, output_diff_xf_cv);
    xf::cv::analyzeDiff(output_diff_xf_cv, 0, err_per);
    cv::imwrite("xf_cv_diff_image.png", output_diff_xf_cv);
    cv::imwrite("xf_image.png", output_xf);
}
