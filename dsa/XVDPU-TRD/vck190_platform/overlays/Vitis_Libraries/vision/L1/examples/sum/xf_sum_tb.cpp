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
#include "xf_sum_config.h"
#include <ap_int.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1>\n");
        return EXIT_FAILURE;
    }

    cv::Mat in_gray, in_gray1, out_gray, diff;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    int channels = in_gray.channels();
    int height = in_gray.rows;
    int width = in_gray.cols;
    // OpenCV function
    std::vector<double> ocv_scl(channels);

    for (int i = 0; i < channels; ++i) ocv_scl[i] = cv::sum(in_gray)[i];

    double* scl = (double*)malloc(channels * sizeof(double));

    // Call the top function
    sum_accel((ap_uint<PTR_WIDTH>*)in_gray.data, scl);

    for (int i = 0; i < in_gray.channels(); i++) {
        printf("sum of opencv is=== %lf\n", ocv_scl[i]);
        printf("sum of hls is====== %lf\n", scl[i]);
    }

    // Results verification:
    int cnt = 0;

    for (int i = 0; i < in_gray.channels(); i++) {
        if (abs(double(ocv_scl[i]) - scl[i]) > 0.01f) cnt++;
    }

    if (cnt > 0) {
        fprintf(stderr, "INFO: Error percentage = %d%. Test Failed.\n ", 100.0 * float(cnt) / float(channels));
        return EXIT_FAILURE;
    } else
        std::cout << "Test Passed." << std::endl;

    return 0;
}
