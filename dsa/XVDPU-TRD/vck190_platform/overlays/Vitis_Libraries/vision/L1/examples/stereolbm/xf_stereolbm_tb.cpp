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
#include "xf_stereolbm_config.h"

using namespace std;

#define _TEXTURE_THRESHOLD_ 20
#define _UNIQUENESS_RATIO_ 15
#define _PRE_FILTER_CAP_ 31
#define _MIN_DISP_ 0

int main(int argc, char** argv) {
    cv::setUseOptimized(false);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat left_img, right_img;

    // Reading in the images: Only Grayscale image
    left_img = cv::imread(argv[1], 0);
    right_img = cv::imread(argv[2], 0);

    if (left_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    if (right_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }

    int rows = left_img.rows;
    int cols = left_img.cols;

    cv::Mat disp, hls_disp;

    // OpenCV reference function:
    /*cv::StereoBM bm;
    bm.state->preFilterCap = _PRE_FILTER_CAP_;
    bm.state->preFilterType = CV_STEREO_BM_XSOBEL;
    bm.state->SADWindowSize = SAD_WINDOW_SIZE;
    bm.state->minDisparity = _MIN_DISP_;
    bm.state->numberOfDisparities = NO_OF_DISPARITIES;
    bm.state->textureThreshold = _TEXTURE_THRESHOLD_;
    bm.state->uniquenessRatio = _UNIQUENESS_RATIO_;
    bm(left_img, right_img, disp);*/

    // enable this if the above code is obsolete
    cv::Ptr<cv::StereoBM> stereobm = cv::StereoBM::create(NO_OF_DISPARITIES, SAD_WINDOW_SIZE);
    stereobm->setPreFilterCap(_PRE_FILTER_CAP_);
    stereobm->setUniquenessRatio(_UNIQUENESS_RATIO_);
    stereobm->setTextureThreshold(_TEXTURE_THRESHOLD_);
    stereobm->compute(left_img, right_img, disp);

    cv::Mat disp8, hls_disp8;
    disp.convertTo(disp8, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));
    cv::imwrite("ocv_output.png", disp8);
    // end of reference

    // Creating host memory for the hw acceleration
    hls_disp.create(rows, cols, CV_16UC1);
    hls_disp8.create(rows, cols, CV_8UC1);

    // OpenCL section:
    std::vector<unsigned char> bm_state_params(4);
    bm_state_params[0] = _PRE_FILTER_CAP_;
    bm_state_params[1] = _UNIQUENESS_RATIO_;
    bm_state_params[2] = _TEXTURE_THRESHOLD_;
    bm_state_params[3] = _MIN_DISP_;

    stereolbm_accel((ap_uint<INPUT_PTR_WIDTH>*)left_img.data, (ap_uint<INPUT_PTR_WIDTH>*)right_img.data,
                    (unsigned char*)bm_state_params.data(), (ap_uint<OUTPUT_PTR_WIDTH>*)hls_disp.data, rows, cols);

    // Convert 16U output to 8U output:
    hls_disp.convertTo(hls_disp8, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));
    cv::imwrite("hls_out.jpg", hls_disp8);

    ////////  FUNCTIONAL VALIDATION  ////////
    // changing the invalid value from negative to zero for validating the
    // difference
    cv::Mat disp_u(rows, cols, CV_16UC1);
    for (int i = 0; i < disp.rows; i++) {
        for (int j = 0; j < disp.cols; j++) {
            if (disp.at<short>(i, j) < 0) {
                disp_u.at<unsigned short>(i, j) = 0;
            } else
                disp_u.at<unsigned short>(i, j) = (unsigned short)disp.at<short>(i, j);
        }
    }

    cv::Mat diff;
    diff.create(left_img.rows, left_img.cols, CV_16UC1);
    cv::absdiff(disp_u, hls_disp, diff);
    cv::imwrite("diff_img.jpg", diff);

    // removing border before diff analysis
    cv::Mat diff_c;
    diff_c.create((diff.rows - SAD_WINDOW_SIZE << 1), diff.cols - (SAD_WINDOW_SIZE << 1), CV_16UC1);
    cv::Rect roi;
    roi.x = SAD_WINDOW_SIZE;
    roi.y = SAD_WINDOW_SIZE;
    roi.width = diff.cols - (SAD_WINDOW_SIZE << 1);
    roi.height = diff.rows - (SAD_WINDOW_SIZE << 1);
    diff_c = diff(roi);

    float err_per;
    xf::cv::analyzeDiff(diff_c, 0, err_per);

    if (err_per > 0.0f) {
        return 1;
    }
    return 0;
}
