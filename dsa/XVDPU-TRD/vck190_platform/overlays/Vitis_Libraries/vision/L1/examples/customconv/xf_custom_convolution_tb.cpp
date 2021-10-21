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
#include "xf_custom_convolution_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, diff, filter;

#if GRAY
    // Reading in the gray image:
    in_img = cv::imread(argv[1], 0);
#else
    // Reading in the gray image:
    in_img = cv::imread(argv[1], 1);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    unsigned char shift = SHIFT;

    // Creating the kernel:
    filter.create(FILTER_HEIGHT, FILTER_WIDTH, CV_32F);

    // Filling the Filter coefficients:
    for (int i = 0; i < FILTER_HEIGHT; i++) {
        for (int j = 0; j < FILTER_WIDTH; j++) {
            filter.at<float>(i, j) = (float)0.1111;
        }
    }

/////////////////    OpenCV reference   /////////////////
#if GRAY
#if OUT_8U
    out_img.create(in_img.rows, in_img.cols, CV_8UC1); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC1);    // create memory for difference image
#elif OUT_16S
    out_img.create(in_img.rows, in_img.cols, CV_16SC1); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_16SC1);    // create memory for difference image
#endif
#else
#if OUT_8U
    out_img.create(in_img.rows, in_img.cols, CV_8UC3); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC3);    // create memory for difference image
#elif OUT_16S
    out_img.create(in_img.rows, in_img.cols, CV_16SC3); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_16SC3);    // create memory for difference image
#endif
#endif

    cv::Point anchor = cv::Point(-1, -1);

#if OUT_8U
    cv::filter2D(in_img, ocv_ref, CV_8U, filter, anchor, 0, cv::BORDER_CONSTANT);
#elif OUT_16S
    cv::filter2D(in_img, ocv_ref, CV_16S, filter, anchor, 0, cv::BORDER_CONSTANT);
#endif

    cv::imwrite("ref_img.jpg", ocv_ref); // reference image

    /*  std::vector<short int> filter_vec(FILTER_WIDTH * FILTER_HEIGHT);

      for (int i = 0; i < FILTER_HEIGHT; i++) {
          for (int j = 0; j < FILTER_WIDTH; j++) {
              filter_vec[i * FILTER_WIDTH + j] = 3640;
          }
      }*/

    short int* filter_ptr = (short int*)malloc(FILTER_WIDTH * FILTER_HEIGHT * sizeof(short int));

    for (int i = 0; i < FILTER_HEIGHT; i++) {
        for (int j = 0; j < FILTER_WIDTH; j++) {
            filter_ptr[i * FILTER_WIDTH + j] = 3640;
        }
    }

    int rows = in_img.rows;
    int cols = in_img.cols;
#if GRAY
    // OpenCL section:
    size_t image_in_size_bytes = in_img.rows * in_img.cols * sizeof(unsigned char);
#else
    // OpenCL section:
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
#endif

#if GRAY
#if OUT_8U == 1
    size_t image_out_size_bytes = in_img.rows * in_img.cols * sizeof(unsigned char);
#else
    size_t image_out_size_bytes = in_img.rows * in_img.cols * sizeof(short int);
#endif
#else
#if OUT_8U == 1
    size_t image_out_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
#else
    size_t image_out_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(short int);
#endif
#endif

    //////////////Top function call /////////////////////////////

    Filter2d_accel((ap_uint<INPUT_PTR_WIDTH>*)in_img.data, filter_ptr, shift, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data,
                   rows, cols);

    // Save the kernel result:
    cv::imwrite("out_img.jpg", out_img);

    // Results verification:
    cv::absdiff(ocv_ref, out_img, diff); // Compute absolute difference image
    cv::imwrite("diff_img.jpg", diff);   // Save the difference image for debugging purpose

    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
#if OUT_8U == 1
            unsigned char v = diff.at<unsigned char>(i, j);
#elif OUT_16S == 1
            short int v = diff.at<short int>(i, j);
#endif
            if (v > 2) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }

    float err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols);

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
