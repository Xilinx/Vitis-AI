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
#include "xf_reduce_config.h"
#include <ap_int.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1>\n");
        return EXIT_FAILURE;
    }

    cv::Mat in_img, dst_hls, ocv_ref, in_gray, diff, in_mask;

    // Reading in the image:
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

#if DIM
    if ((REDUCTION_OP == cv::REDUCE_AVG) || (REDUCTION_OP == cv::REDUCE_SUM)) {
        dst_hls.create(in_img.rows, 1, CV_32SC1);
        ocv_ref.create(in_img.rows, 1, CV_32SC1);
    } else {
        dst_hls.create(in_img.rows, 1, CV_8UC1);
        ocv_ref.create(in_img.rows, 1, CV_8UC1);
    }

#else
    if ((REDUCTION_OP == cv::REDUCE_AVG) || (REDUCTION_OP == cv::REDUCE_SUM)) {
        dst_hls.create(1, in_img.cols, CV_32SC1);
        ocv_ref.create(1, in_img.cols, CV_32SC1);
    } else {
        dst_hls.create(1, in_img.cols, CV_8UC1);
        ocv_ref.create(1, in_img.cols, CV_8UC1);
    }
#endif

    int height = in_img.rows;
    int width = in_img.cols;
    unsigned char dimension = DIM;
    size_t image_out_size_bytes;

    // Call the top function
    reduce_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, dimension, (ap_uint<OUTPUT_PTR_WIDTH>*)dst_hls.data, height,
                 width);

    // Reference function
    if ((REDUCTION_OP == cv::REDUCE_AVG) || (REDUCTION_OP == cv::REDUCE_SUM))
        cv::reduce(in_img, ocv_ref, DIM, REDUCTION_OP, CV_32SC1); // avg, sum
    else
        cv::reduce(in_img, ocv_ref, DIM, REDUCTION_OP, CV_8UC1);

    // Results verification:
    FILE* fp = fopen("hls", "w");
    FILE* fp1 = fopen("cv", "w");
    int err_cnt = 0;

#if DIM == 1
    for (unsigned int i = 0; i < dst_hls.rows; i++) {
        fprintf(fp, "%d\n", (unsigned char)dst_hls.data[i]);
        fprintf(fp1, "%d\n", ocv_ref.data[i]);
        unsigned int diff = ocv_ref.data[i] - (unsigned char)dst_hls.data[i];
        if (diff > 1) err_cnt++;
    }

    std::cout << "INFO: Percentage of pixels with an error = " << (float)err_cnt * 100 / (float)dst_hls.rows << "%"
              << std::endl;

#endif
#if DIM == 0
    for (int i = 0; i < dst_hls.cols; i++) {
        fprintf(fp, "%d\n", (unsigned char)dst_hls.data[i]);
        fprintf(fp1, "%d\n", ocv_ref.data[i]);
        unsigned int diff = ocv_ref.data[i] - (unsigned char)dst_hls.data[i];
        if (diff > 1) err_cnt++;
    }

    std::cout << "INFO: Percentage of pixels with an error = " << (float)err_cnt * 100 / (float)dst_hls.cols << "%"
              << std::endl;

#endif
    fclose(fp);
    fclose(fp1);

    if (err_cnt > 0) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
