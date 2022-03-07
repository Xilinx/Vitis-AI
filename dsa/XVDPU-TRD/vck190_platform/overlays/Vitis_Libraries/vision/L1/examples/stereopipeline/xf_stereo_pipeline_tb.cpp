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
#include "xf_stereo_pipeline_config.h"
#include "cameraParameters.h"

#define _PROFILE_ 0

using namespace std;

int main(int argc, char** argv) {
    cv::setUseOptimized(false);

    if (argc != 3) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage: <executable> <left image> <right image>\n");
        return -1;
    }

    cv::Mat left_img, right_img;
    left_img = cv::imread(argv[1], 0);
    if (left_img.data == NULL) {
        fprintf(stderr, "Cannot open left image at %s\n", argv[1]);
        return 0;
    }
    right_img = cv::imread(argv[2], 0);
    if (right_img.data == NULL) {
        fprintf(stderr, "Cannot open right image at %s\n", argv[1]);
        return 0;
    }

    //////////////////	HLS TOP Function Call  ////////////////////////
    int rows = left_img.rows;
    int cols = left_img.cols;
    cv::Mat disp_img(rows, cols, CV_16UC1);

    // allocate mem for camera parameters for rectification and bm_state class
    float* cameraMA_l_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* cameraMA_r_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* irA_l_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* irA_r_fl = (float*)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(float));
    float* distC_l_fl = (float*)malloc(XF_DIST_COEFF_SIZE * sizeof(float));
    float* distC_r_fl = (float*)malloc(XF_DIST_COEFF_SIZE * sizeof(float));
    int* bm_state_arr = (int*)malloc(11 * sizeof(int));

    xf::cv::xFSBMState<SAD_WINDOW_SIZE, NO_OF_DISPARITIES, PARALLEL_UNITS> bm_state;
    bm_state.uniquenessRatio = 15;
    bm_state.textureThreshold = 20;
    bm_state.minDisparity = 0;
    bm_state_arr[0] = bm_state.preFilterType;
    bm_state_arr[1] = bm_state.preFilterSize;
    bm_state_arr[2] = bm_state.preFilterCap;
    bm_state_arr[3] = bm_state.SADWindowSize;
    bm_state_arr[4] = bm_state.minDisparity;
    bm_state_arr[5] = bm_state.numberOfDisparities;
    bm_state_arr[6] = bm_state.textureThreshold;
    bm_state_arr[7] = bm_state.uniquenessRatio;
    bm_state_arr[8] = bm_state.ndisp_unit;
    bm_state_arr[9] = bm_state.sweepFactor;
    bm_state_arr[10] = bm_state.remainder;

    // copy camera params
    for (int i = 0; i < XF_CAMERA_MATRIX_SIZE; i++) {
        cameraMA_l_fl[i] = (float)cameraMA_l[i];
        cameraMA_r_fl[i] = (float)cameraMA_r[i];
        irA_l_fl[i] = (float)irA_l[i];
        irA_r_fl[i] = (float)irA_r[i];
    }

    // copy distortion coefficients
    for (int i = 0; i < XF_DIST_COEFF_SIZE; i++) {
        distC_l_fl[i] = (float)distC_l[i];
        distC_r_fl[i] = (float)distC_r[i];
    }

    // Launch the kernel
    stereopipeline_accel((ap_uint<INPUT_PTR_WIDTH>*)left_img.data, (ap_uint<INPUT_PTR_WIDTH>*)right_img.data,
                         (ap_uint<OUTPUT_PTR_WIDTH>*)disp_img.data, cameraMA_l_fl, cameraMA_r_fl, distC_l_fl,
                         distC_r_fl, irA_l_fl, irA_r_fl, bm_state_arr, rows, cols);

    // Write output image
    cv::Mat out_disp_img(rows, cols, CV_8UC1);
    disp_img.convertTo(out_disp_img, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));
    cv::imwrite("hls_output.png", out_disp_img);
    printf("run complete !\n");

    return 0;
}
