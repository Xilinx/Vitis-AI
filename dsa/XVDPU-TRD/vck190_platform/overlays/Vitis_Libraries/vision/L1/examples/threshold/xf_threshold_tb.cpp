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
#include "xf_threshold_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_ref, in_gray, diff;

    unsigned short in_width, in_height;

    /*  reading in the color image  */
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    in_width = in_img.cols;
    in_height = in_img.rows;

    ocv_ref.create(in_img.rows, in_img.cols, in_img.depth());
    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    diff.create(in_img.rows, in_img.cols, in_img.depth());

    ////////////////  reference code  ////////////////

    unsigned char maxval = 50;
    unsigned char thresh = 100;

    cv::threshold(in_img, ocv_ref, thresh, maxval, THRESH_TYPE);
    //////////////////  end opencv reference code//////////

    /////////////// Call the top function ////////////////
    threshold_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, thresh, maxval,
                    in_height, in_width);

    // Write output image
    imwrite("hls_out.jpg", out_img);

    // Compute absolute difference image
    absdiff(ocv_ref, out_img, diff);

    // Save the difference image
    imwrite("diff.png", diff);

    float err_per;
    xf::cv::analyzeDiff(diff, 0, err_per);

    if (err_per > 0.0f) {
        return 1;
    }

    return 0;
}
