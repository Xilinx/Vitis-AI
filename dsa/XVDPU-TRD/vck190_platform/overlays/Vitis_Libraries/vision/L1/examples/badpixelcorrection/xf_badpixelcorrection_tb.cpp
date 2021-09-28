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
#include "xf_badpixelcorrection_config.h"

#include <time.h>
void BadPixelCorrection(cv::Mat input, cv::Mat& output) {
#if T_8U
    typedef unsigned char Pixel_t;
#else
    typedef unsigned short int Pixel_t;
#endif
    const Pixel_t MINVAL = 0;
    const Pixel_t MAXVAL = -1;
    cv::Mat mask =
        (cv::Mat_<unsigned char>(5, 5) << 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1);
    output = input.clone(); // Not cloning saves memory
    cv::Mat min, max;
    cv::erode(input, min, mask, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, MAXVAL);  // Min Filter
    cv::dilate(input, max, mask, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, MINVAL); // Max Filter

    cv::subtract(min, input, min);                    // Difference of min and input
    cv::subtract(input, max, max);                    // Difference of input and max
    cv::threshold(min, min, 0, 0, cv::THRESH_TOZERO); // Remove all values less than zero (not required for this case
                                                      // but might be required for other data types which have signed
                                                      // values)
    cv::threshold(max, max, 0, 0, cv::THRESH_TOZERO); // Remove all values less than zero
    cv::subtract(output, max, output);
    cv::add(output, min, output);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path1> \n");
        return -1;
    }

    cv::Mat in_gray, in_gray1, ocv_ref, out_gray, diff, ocv_ref_in1, ocv_ref_in2, inout_gray1, ocv_ref_gw;
#if T_8U
    in_gray = cv::imread(argv[1], 0); // read image
#else
    in_gray = cv::imread(argv[1], -1); // read image
#endif
    if (in_gray.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[1]);
        return -1;
    }

    ocv_ref.create(in_gray.rows, in_gray.cols, in_gray.type());
    ocv_ref_gw.create(in_gray.rows, in_gray.cols, in_gray.type());
    out_gray.create(in_gray.rows, in_gray.cols, in_gray.type());
    diff.create(in_gray.rows, in_gray.cols, in_gray.type());

    // OpenCV Reference
    BadPixelCorrection(in_gray, ocv_ref);

    /////////////////////////////////////// CL ////////////////////////

    int height = in_gray.rows;
    int width = in_gray.cols;

    //////////////////////////Top function call ///////////////////////
    badpixelcorrection_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray.data, (ap_uint<INPUT_PTR_WIDTH>*)out_gray.data, height,
                             width);

    imwrite("out_hls.jpg", out_gray);
    imwrite("ocv_ref.png", ocv_ref);

    cv::absdiff(ocv_ref, out_gray, diff);
    imwrite("error.png", diff); // Save the difference image for debugging purpose

    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    if (err_per > 1) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    } else
        std::cout << "Test Passed " << std::endl;

    return 0;
}
