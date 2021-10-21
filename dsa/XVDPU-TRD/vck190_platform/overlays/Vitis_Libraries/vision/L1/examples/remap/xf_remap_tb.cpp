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
#include "xf_remap_config.h"

#define READ_MAPS_FROM_FILE 0

int main(int argc, char** argv) {
#if READ_MAPS_FROM_FILE
    if (argc != 4) {
        fprintf(stderr, "Usage: <executable> <input image path> <mapx file> <mapy file>\n");
        return -1;
    }
#else
    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image path>\n");
        return -1;
    }
#endif

    cv::Mat src, ocv_remapped, hls_remapped;
    cv::Mat map_x, map_y, diff;

// Reading in the image:
#if GRAY
    src = cv::imread(argv[1], 0); // read image Grayscale
#else
    src = cv::imread(argv[1], 1); // read image RGB
#endif

    if (!src.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Allocate memory for the outputs:
    std::cout << "INFO: Allocate memory for input and output data." << std::endl;
    ocv_remapped.create(src.rows, src.cols, src.type()); // opencv result
    map_x.create(src.rows, src.cols, CV_32FC1);          // Mapx for opencv remap function
    map_y.create(src.rows, src.cols, CV_32FC1);          // Mapy for opencv remap function
    hls_remapped.create(src.rows, src.cols, src.type()); // create memory for output images
    diff.create(src.rows, src.cols, src.type());

// Initialize the float maps:
#if READ_MAPS_FROM_FILE
    // read the float map data from the file (code could be alternated for reading
    // from image)
    FILE *fp_mx, *fp_my;
    fp_mx = fopen(argv[2], "r");
    fp_my = fopen(argv[3], "r");
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float valx, valy;
            if (fscanf(fp_mx, "%f", &valx) != 1) {
                fprintf(stderr, "Not enough data in the provided map_x file ... !!!\n ");
            }
            if (fscanf(fp_my, "%f", &valy) != 1) {
                fprintf(stderr, "Not enough data in the provided map_y file ... !!!\n ");
            }
            map_x.at<float>(i, j) = valx;
            map_y.at<float>(i, j) = valy;
        }
    }
#else // example map generation, flips the image horizontally
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float valx = (float)(src.cols - j - 1), valy = (float)i;
            map_x.at<float>(i, j) = valx;
            map_y.at<float>(i, j) = valy;
        }
    }
#endif

    // Opencv reference:
    std::cout << "INFO: Run reference function in CV." << std::endl;
#if INTERPOLATION == 0
    cv::remap(src, ocv_remapped, map_x, map_y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
#else
    cv::remap(src, ocv_remapped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
#endif

    remap_accel((ap_uint<PTR_IMG_WIDTH>*)src.data, (float*)map_x.data, (float*)map_y.data,
                (ap_uint<PTR_IMG_WIDTH>*)hls_remapped.data, src.rows, src.cols);

    // Save the results:
    cv::imwrite("ocv_reference_out.jpg", ocv_remapped); // Opencv Result
    cv::imwrite("kernel_out.jpg", hls_remapped);

    // Results verification:
    cv::absdiff(ocv_remapped, hls_remapped, diff);
    cv::imwrite("diff.png", diff);

    // Find minimum and maximum differences.
    float err_per;
    xf::cv::analyzeDiff(diff, 0, err_per);

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
