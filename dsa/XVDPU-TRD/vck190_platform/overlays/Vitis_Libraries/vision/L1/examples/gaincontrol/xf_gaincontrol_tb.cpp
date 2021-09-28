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
#include "xf_gaincontrol_config.h"

#include <time.h>
void gainControlOCV(cv::Mat input, cv::Mat& output, int code, unsigned short rgain, unsigned short bgain) {
    cv::Mat mat = input.clone();
    int height = mat.size().height;
    int width = mat.size().width;
#if T_8U
    typedef uint8_t realSize;
#else
    typedef uint16_t realSize;
#endif
    typedef unsigned int maxSize;
    maxSize pixel;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // std::cout<<"("<<i<<","<<j<<")\t";
            pixel = (maxSize)mat.at<realSize>(i, j); // extracting each pixel
            // std::cout<<"Initial: "<<pixel<<"\t";
            bool cond1, cond2;
            cond1 = (j % 2 == 0);
            cond2 = (j % 2 != 0);
            if (code == XF_BAYER_RG) {
                if (i % 2 == 0 && cond1)
                    pixel = (maxSize)((pixel * rgain) >> 7);
                else if (i % 2 != 0 && cond2)
                    pixel = (maxSize)((pixel * bgain) >> 7);
            } else if (code == XF_BAYER_GR) {
                if (i % 2 == 0 && cond2)
                    pixel = (maxSize)((pixel * rgain) >> 7);
                else if (i % 2 != 0 && cond1)
                    pixel = (maxSize)((pixel * bgain) >> 7);
            } else if (code == XF_BAYER_BG) {
                if (i % 2 == 0 && cond1)
                    pixel = (maxSize)((pixel * bgain) >> 7);
                else if (i % 2 == 0 && cond2)
                    pixel = (maxSize)((pixel * rgain) >> 7);
            } else if (code == XF_BAYER_GB) {
                if (i % 2 == 0 && cond2)
                    pixel = (maxSize)((pixel * bgain) >> 7);
                else if (i % 2 != 0 && cond1)
                    pixel = (maxSize)((pixel * rgain) >> 7);
            }
            // std::cout<<"Final: "<<pixel<<std::endl;
            mat.at<realSize>(i, j) = cv::saturate_cast<realSize>(pixel); // writing each pixel
        }
    }
    output = mat;
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

    int height = in_gray.rows;
    int width = in_gray.cols;
    unsigned short rgain = 154;
    unsigned short bgain = 140;

    // OpenCV Reference
    gainControlOCV(in_gray, ocv_ref, BFORMAT, rgain, bgain);

    ///////////////////////////Top function call ///////////////////////////

    gaincontrol_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_gray.data, height, width,
                      rgain, bgain);

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
