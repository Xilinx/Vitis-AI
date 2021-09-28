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
#include "opencv2/features2d.hpp"

#include "xf_distancetransform_config.h"

#include "xcl2.hpp"
#include "xf_opencl_wrap.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_ref;

    // reading in the image
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    /////////////////////	  OpenCV reference	 ////////////////
    cv::distanceTransform(in_img, ocv_ref, cv::DIST_L2, 3);
    cv::imwrite("out_ocv.jpg", ocv_ref);
    /////////////////////	End of OpenCV reference	 ////////////////

    out_img.create(in_img.rows, in_img.cols, ocv_ref.type());
    int height = in_img.rows;
    int width = in_img.cols;
    unsigned int* fw_pass_data = (unsigned int*)malloc(height * width * 4); // 4-bytes

    ////////////////////	HLS TOP function call	/////////////////

    /////////////////////////////////////// CL ////////////////////////
    (void)cl_kernel_mgr::registerKernel("distancetransform_accel", "krnl_distancetransform", XCLIN(in_img),
                                        XCLOUT(out_img), XCLIN(fw_pass_data, height * width * 4), XCLIN(height),
                                        XCLIN(width));
    cl_kernel_mgr::exec_all();
    /////////////////////////////////////// end of CL ////////////////////////

    // Write output image
    cv::imwrite("hw_out.jpg", out_img);

    //////////////////  Validation ////////////////////
    int cnt = 0;
    for (int i = 0; i < out_img.rows; i++) {
        for (int j = 0; j < out_img.cols; j++) {
            float a = (out_img.at<float>(i, j) - ocv_ref.at<float>(i, j));
            if (a < 0) a = -a;
            if (a > 0.01f) {
                printf("missmatches at: row:%d col:%d hw_data:%5f ocv_ref_output:%5f\n", i, j, out_img.at<float>(i, j),
                       ocv_ref.at<float>(i, j));
                cnt++;
            }
        }
    }
    if (cnt > 0)
        printf("test failed!\nNo. of missmatched outputs:%d\n", cnt);
    else
        printf("test passed!\n");

    return 0;
}
