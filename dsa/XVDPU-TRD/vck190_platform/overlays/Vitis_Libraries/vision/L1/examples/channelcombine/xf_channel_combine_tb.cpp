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
#include "xf_channel_combine_config.h"

int main(int argc, char** argv) {
#if FOUR_INPUT
    if (argc != 5) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr,
                "<Executable Name> <input image1 path> <input image2 path> <input image3 path> <input image4 path>\n");
        return -1;
    }
#endif
#if THREE_INPUT
    if (argc != 4) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image1 path> <input image2 path> <input image3 path> \n");
        return -1;
    }
#endif
#if TWO_INPUT
    if (argc != 3) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image1 path> <input image2 path> \n");
        return -1;
    }
#endif

    cv::Mat in_gray1, in_gray2;
    cv::Mat in_gray3, in_gray4;
    cv::Mat out_img, ocv_ref;
    cv::Mat diff;

    // Reading in the images:
    in_gray1 = cv::imread(argv[1], 0);
    in_gray2 = cv::imread(argv[2], 0);

    if ((in_gray1.data == NULL) || (in_gray2.data == NULL)) {
        fprintf(stderr, "Cannot open image 4\n");
        return 1;
    }

#if !TWO_INPUT
    in_gray3 = cv::imread(argv[3], 0);
    if ((in_gray3.data == NULL)) {
        fprintf(stderr, "Cannot open input images \n");
        return 1;
    }
    // creating memory for diff image
    diff.create(in_gray1.rows, in_gray1.cols, CV_TYPE);
#endif

#if FOUR_INPUT

    in_gray4 = cv::imread(argv[4], 0);

    if ((in_gray4.data == NULL)) {
        fprintf(stderr, "Cannot open image 4\n");
        return 1;
    }

#endif

    // image height and width
    int height = in_gray1.rows;
    int width = in_gray1.cols;

// Allocate memory for the output images:
#if TWO_INPUT
    out_img.create(in_gray1.rows, in_gray1.cols, CV_8UC2);
#endif
#if THREE_INPUT
    out_img.create(in_gray1.rows, in_gray1.cols, CV_8UC3);
#endif
#if FOUR_INPUT
    out_img.create(in_gray1.rows, in_gray1.cols, CV_8UC4);
#endif

// Call the top function
#if TWO_INPUT
    channel_combine_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray1.data, (ap_uint<INPUT_PTR_WIDTH>*)in_gray2.data,
                          (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, height, width);
#endif
#if THREE_INPUT
    channel_combine_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray1.data, (ap_uint<INPUT_PTR_WIDTH>*)in_gray2.data,
                          (ap_uint<INPUT_PTR_WIDTH>*)in_gray3.data, (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, height,
                          width);
#endif
#if FOUR_INPUT
    channel_combine_accel((ap_uint<INPUT_PTR_WIDTH>*)in_gray1.data, (ap_uint<INPUT_PTR_WIDTH>*)in_gray2.data,
                          (ap_uint<INPUT_PTR_WIDTH>*)in_gray3.data, (ap_uint<INPUT_PTR_WIDTH>*)in_gray4.data,
                          (ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data, height, width);
#endif

#if !TWO_INPUT
    // Write the kernel output image:
    cv::imwrite("hls_out.jpg", out_img);

    // OpenCV reference:
    std::vector<cv::Mat> bgr_planes;
    cv::Mat merged;

    bgr_planes.push_back(in_gray1);
    bgr_planes.push_back(in_gray2);

    bgr_planes.push_back(in_gray3);

#if FOUR_INPUT
    bgr_planes.push_back(in_gray4);
#endif

    cv::merge(bgr_planes, merged);

    // Results verification:
    cv::imwrite("out_ocv.jpg", merged);
    cv::absdiff(merged, out_img, diff);
    cv::imwrite("diff.jpg", diff);

    // Find minimum and maximum differences:
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
#if FOUR_INPUT
            cv::Vec4b v = diff.at<cv::Vec4b>(i, j);
#endif
#if THREE_INPUT
            cv::Vec3b v = diff.at<cv::Vec3b>(i, j);
#endif
            if (v[0] > 0) cnt++;
            if (v[1] > 0) cnt++;
            if (v[2] > 0) cnt++;
#if FOUR_INPUT
            if (v[3] > 0) cnt++;
#endif
            if (minval > v[0]) minval = v[0];
            if (minval > v[1]) minval = v[1];
            if (minval > v[2]) minval = v[2];
#if FOUR_INPUT
            if (minval > v[3]) minval = v[3];
#endif
            if (maxval < v[0]) maxval = v[0];
            if (maxval < v[1]) maxval = v[1];
            if (maxval < v[2]) maxval = v[2];
#if FOUR_INPUT
            if (maxval < v[3]) maxval = v[3];
#endif
        }
    }

    float err_per = 100.0 * (float)cnt / (out_img.rows * out_img.cols * out_img.channels());

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << "%" << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }
#endif
    return 0;
}
