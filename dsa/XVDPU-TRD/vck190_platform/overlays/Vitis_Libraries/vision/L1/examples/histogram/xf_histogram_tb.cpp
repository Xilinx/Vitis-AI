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
#include "xf_histogram_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_gray, hist_ocv;

#if GRAY
    // reading in the color image
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }
    // cvtColor(in_img, in_img, CV_BGR2GRAY);
    //////////////////	Opencv Reference  ////////////////////////
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&in_img, 1, 0, cv::Mat(), hist_ocv, 1, &histSize, &histRange, 1, 0);

#else
    // reading in the color image
    in_img = cv::imread(argv[1], 1);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }
    //////////////////	Opencv Reference  ////////////////////////
    cv::Mat b_hist, g_hist, r_hist;
    std::vector<cv::Mat> bgr_planes;
    cv::split(in_img, bgr_planes);
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float* histRange[] = {range};
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, 1, 0);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, 1, 0);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, 1, 0);
#endif

    // Create a memory to hold HLS implementation output:

    unsigned int* histogram = (unsigned int*)malloc(histSize * in_img.channels() * sizeof(unsigned int));

    int rows = in_img.rows;
    int cols = in_img.cols;

    //////////////// Top function call ///////////////////////
    histogram_accel((ap_uint<PTR_WIDTH>*)in_img.data, histogram, rows, cols);

#if GRAY
    FILE *fp, *fp1;
    fp = fopen("out_hls.txt", "w");
    fp1 = fopen("out_ocv.txt", "w");
    for (int cnt = 0; cnt < 256; cnt++) {
        fprintf(fp, "%u\n", histogram[cnt]);
        uint32_t val = (uint32_t)hist_ocv.at<float>(cnt);
        if (val != histogram[cnt]) {
            fprintf(stderr, "Test Failed.\n ");
            return 1;
        }
        fprintf(fp1, "%u\n", val);
    }
    fclose(fp);
    fclose(fp1);
#else
    FILE* total = fopen("total.txt", "w");
    for (int i = 0; i < 768; i++) {
        fprintf(total, "%d\n", histogram[i]);
    }
    fclose(total);
    FILE *fp, *fp1;
    fp = fopen("out_hls.txt", "w");
    fp1 = fopen("out_ocv.txt", "w");
    for (int cnt = 0; cnt < 256; cnt++) {
        fprintf(fp, "%u	%u	%u\n", histogram[cnt], histogram[cnt + 256], histogram[cnt + 512]);
        uint32_t b_val = (uint32_t)b_hist.at<float>(cnt);
        uint32_t g_val = (uint32_t)g_hist.at<float>(cnt);
        uint32_t r_val = (uint32_t)r_hist.at<float>(cnt);
        if ((b_val != histogram[cnt]) && (g_val != histogram[256 + cnt]) && (r_val != histogram[512 + cnt])) {
            fprintf(stderr, "ERROR: Test Failed.\n ");
            return 1;
        }
        fprintf(fp1, "%u	%u	%u\n", b_val, g_val, r_val);
    }
    fclose(fp);
    fclose(fp1);
#endif

    return 0;
}
