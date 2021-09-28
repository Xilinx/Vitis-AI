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
#include "xf_boundingbox_config.h"

#include <sys/time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

using namespace std;

int main(int argc, char** argv) {
    cv::Mat in_img, in_img1, out_img, diff;

    struct timespec start_time;
    struct timespec end_time;

    if (argc != 3) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> <number of boxes>\n");
        return -1;
    }

#if GRAY
    /*  reading in the gray image  */
    in_img = cv::imread(argv[1], 0);
    in_img1 = in_img.clone();
    int num_box = atoi(argv[2]);
#else
    /*  reading in the color image  */
    in_img = cv::imread(argv[1], 1);
    cvtColor(in_img, in_img, cv::COLOR_BGR2RGBA);
    in_img1 = in_img.clone();
    int num_box = atoi(argv[2]);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }
    unsigned int x_loc[MAX_BOXES], y_loc[MAX_BOXES], ROI_height[MAX_BOXES], ROI_width[MAX_BOXES];

    /////////////////////////////////////Feeding ROI/////////////////////////////////////////

    x_loc[0] = 0; // only 3-ROI are feeded, should be modified according to NUM_BOX
    y_loc[0] = 0;
    ROI_height[0] = 48;
    ROI_width[0] = 64;

    x_loc[1] = 0;
    y_loc[1] = 0;
    ROI_height[1] = 100;
    ROI_width[1] = 120;

    x_loc[2] = 50;
    y_loc[2] = 50;
    ROI_height[2] = 30;
    ROI_width[2] = 30;

    x_loc[3] = 45;
    y_loc[3] = 45;
    ROI_height[3] = 67;
    ROI_width[3] = 67;

    x_loc[4] = 67;
    y_loc[4] = 67;
    ROI_height[4] = 10;
    ROI_width[4] = 10;

//////////////////////////////////end of Feeding ROI///////////////////////////////////////
#if GRAY
    int color_info[MAX_BOXES][4] = {{255, 0, 0, 0}, {110, 0, 0, 0}, {0, 0, 0, 0}, {150, 0, 0, 0}, {56, 0, 0, 0}};
#else
    int color_info[MAX_BOXES][4] = {
        {255, 0, 0, 255},
        {0, 255, 0, 255},
        {0, 0, 255, 255},
        {123, 234, 108, 255},
        {122, 255, 167, 255}}; // Feeding color information for each boundary should be modified if MAX_BOXES varies
#endif

#if GRAY
    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    diff.create(in_img.rows, in_img.cols, in_img.depth());

#else
    diff.create(in_img.rows, in_img.cols, CV_8UC4);
    out_img.create(in_img.rows, in_img.cols, CV_8UC4);
#endif

    ////////////////  reference code  ////////////////
    clock_gettime(CLOCK_MONOTONIC, &start_time);

#if GRAY
    for (int i = 0; i < num_box; i++) {
        for (int c = 0; c < XF_CHANNELS(TYPE, NPIX); c++) {
            cv::rectangle(in_img1, cv::Rect(x_loc[i], y_loc[i], ROI_width[i], ROI_height[i]),
                          cv::Scalar(color_info[i][0], 0, 0), 1); // BGR format
        }
    }
#else
    for (int i = 0; i < num_box; i++) {
        for (int c = 0; c < XF_CHANNELS(TYPE, NPIX); c++) {
            cv::rectangle(in_img1, cv::Rect(x_loc[i], y_loc[i], ROI_width[i], ROI_height[i]),
                          cv::Scalar(color_info[i][0], color_info[i][1], color_info[i][2], 255), 1); // BGR format
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float diff_latency = (end_time.tv_nsec - start_time.tv_nsec) / 1e9 + end_time.tv_sec - start_time.tv_sec;
    printf("\nReference code latency: %f ", diff_latency);

    cv::imwrite("ocv_ref.jpg", in_img1); // reference image

    //////////////////  end opencv reference code//////////

    int* roi = (int*)malloc(MAX_BOXES * 4 * sizeof(int));
    //	ap_uint<32> *color=(ap_uint<32>*)malloc(MAX_BOXES*sizeof(ap_uint<32>));

    for (int i = 0, j = 0; i < (MAX_BOXES * 4); j++, i += 4) {
        roi[i] = x_loc[j];
        roi[i + 1] = y_loc[j];
        roi[i + 2] = ROI_height[j];
        roi[i + 3] = ROI_width[j];
    }

    /*		for(int i=0;i<(MAX_BOXES);i++)
                    {

                            for(int j=0,k=0;j<XF_CHANNELS(TYPE,NPIX);j++,k+=XF_DTPIXELDEPTH(TYPE,NPIX))
                            {
                                    color[i].range(k+(XF_DTPIXELDEPTH(TYPE,NPIX)-1),k)  = color_info[i][j];
                            }
                    }*/
    int height = in_img.rows;
    int width = in_img.cols;

    //////////////// Call the top function ////////////////
    boundingbox_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, roi, color_info, height, width, num_box);

    cv::imwrite("hls_out.jpg", in_img);

    cv::absdiff(in_img, in_img1, diff);
    cv::imwrite("diff.jpg", diff); // Save the difference image for debugging purpose

    //	 Find minimum and maximum differences.

    double minval = 256, maxval1 = 0;
    int cnt = 0;
    for (int i = 0; i < in_img1.rows; i++) {
        for (int j = 0; j < in_img1.cols; j++) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 1) cnt++;
            if (minval > v) minval = v;
            if (maxval1 < v) maxval1 = v;
        }
    }
    float err_per = 100.0 * (float)cnt / (in_img1.rows * in_img1.cols);

    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval1 << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return 1;
    }

    std::cout << "Test Passed " << std::endl;

    return 0;
}
