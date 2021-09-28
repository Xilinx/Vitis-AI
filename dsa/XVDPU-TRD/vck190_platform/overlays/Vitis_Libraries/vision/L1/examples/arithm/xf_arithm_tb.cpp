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
#include <stdlib.h>
#include <ap_int.h>
#include "xf_arithm_config.h"

int main(int argc, char** argv) {
#if ARRAY
    if (argc != 3) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n");
        return EXIT_FAILURE;
    }
#else
    if (argc != 2) {
        fprintf(stderr, "Usage: <INPUT IMAGE PATH 1>\n");
        return EXIT_FAILURE;
    }

#endif
    cv::Mat in_img1, in_img2, in_gray1, in_gray2, out_img, ocv_ref, diff;

#if GRAY
    // Reading in the image:
    in_gray1 = cv::imread(argv[1], 0);

    if (in_gray1.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
#else
    in_gray1 = cv::imread(argv[1], 1);

    if (in_gray1.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
#endif
#if ARRAY
#if GRAY
    in_gray2 = cv::imread(argv[2], 0);

    if (in_gray2.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }
#else
    in_gray2 = cv::imread(argv[2], 1);

    if (in_gray2.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }

#endif
#endif

    int height = in_gray1.rows;
    int width = in_gray1.cols;

#if SCALAR
    unsigned char scalar[XF_CHANNELS(TYPE, NPC1)];

    for (int i = 0; i < in_gray1.channels(); ++i) {
        scalar[i] = 150;
    }

    size_t vec_in_size_bytes = in_gray1.channels() * sizeof(unsigned char);
#endif

#if GRAY
#if T_16S
    /*  convert to 16S type  */
    in_gray1.convertTo(in_gray1, CV_16SC1);
    in_gray2.convertTo(in_gray2, CV_16SC1);
    out_img.create(in_gray1.rows, in_gray1.cols, CV_16SC1);
    ocv_ref.create(in_gray2.rows, in_gray1.cols, CV_16SC1);
    diff.create(in_gray1.rows, in_gray1.cols, CV_16SC1);
#else
    out_img.create(in_gray1.rows, in_gray1.cols, CV_8UC1);
    ocv_ref.create(in_gray2.rows, in_gray1.cols, CV_8UC1);
    diff.create(in_gray1.rows, in_gray1.cols, CV_8UC1);
#endif
#else
#if T_16S
    /*  convert to 16S type  */
    in_gray1.convertTo(in_gray1, CV_16SC3);
    in_gray2.convertTo(in_gray2, CV_16SC3);
    out_img.create(in_gray1.rows, in_gray1.cols, CV_16SC3);
    ocv_ref.create(in_gray2.rows, in_gray1.cols, CV_16SC3);
    diff.create(in_gray1.rows, in_gray1.cols, CV_16SC3);
#else
    out_img.create(in_gray1.rows, in_gray1.cols, CV_8UC3);
    ocv_ref.create(in_gray2.rows, in_gray1.cols, CV_8UC3);
    diff.create(in_gray1.rows, in_gray1.cols, CV_8UC3);
#endif
#endif

#ifdef FUNCT_MULTIPLY
    float scale = 0.05;
#endif

#if ARRAY
    arithm_accel((ap_uint<PTR_WIDTH>*)in_gray1.data, (ap_uint<PTR_WIDTH>*)in_gray2.data,
#ifdef FUNCT_MULTIPLY
                 scale,
#endif
                 (ap_uint<PTR_WIDTH>*)out_img.data, height, width);

    // Write down the kernel result:
    cv::imwrite("hls_out.jpg", out_img);
#else
    arithm_accel((ap_uint<PTR_WIDTH>*)in_gray1.data, scalar, (ap_uint<PTR_WIDTH>*)out_img.data, height, width);

#endif
    printf("cv_referencestarted\n");

/* OpenCV reference function */
#if ARRAY
#if defined(FUNCT_BITWISENOT)
    cv::CV_FUNCT_NAME(in_gray1, ocv_ref);
#elif defined(FUNCT_ZERO)
    ocv_ref = cv::Mat::zeros(in_gray1.rows, in_gray1.cols, in_gray1.depth());
#else
    cv::CV_FUNCT_NAME(in_gray1, in_gray2, ocv_ref
#ifdef FUNCT_MULTIPLY
                      ,
                      scale
#endif
#ifdef FUNCT_COMPARE
                      ,
                      CV_EXTRA_ARG
#endif
                      );
#endif
#endif

#if SCALAR
#if defined(FUNCT_SET)
    ocv_ref.setTo(cv::Scalar(scalar[0]));
#else
#ifdef FUNCT_SUBRS
    cv::CV_FUNCT_NAME(scalar[0], in_gray1, ocv_ref);
#else
    cv::CV_FUNCT_NAME(in_gray1, scalar[0], ocv_ref
#ifdef FUNCT_COMPARE
                      ,
                      CV_EXTRA_ARG
#endif
                      );
#endif
#endif
#endif

    // Write down the OpenCV outputs:
    cv::imwrite("ref_img.jpg", ocv_ref);

    /* Results verification */
    // Do the diff and save it:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff_img.jpg", diff);

    // Find the percentage of pixels above error threshold:
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray1.rows; i++) {
        for (int j = 0; j < in_gray1.cols; j++) {
            uchar v = diff.at<uchar>(i, j);

            if (v > 2) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }

    float err_per = 100.0 * (float)cnt / (in_gray1.rows * in_gray1.cols);

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
