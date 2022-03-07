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
#include "xf_phase_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_gray, c_grad_x, c_grad_y, c_grad_x1, c_grad_y1, ocv_ref, out_img, diff;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int filter_size = 3;

    /*  reading in the color image  */
    in_img = cv::imread(argv[1], 1);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    /*  convert to gray  */
    cvtColor(in_img, in_gray, cv::COLOR_BGR2GRAY);

    /////////	OpenCV Phase computation API    ///////
    cv::Sobel(in_gray, c_grad_x1, CV_32FC1, 1, 0, filter_size, scale, delta, cv::BORDER_CONSTANT);
    cv::Sobel(in_gray, c_grad_y1, CV_32FC1, 0, 1, filter_size, scale, delta, cv::BORDER_CONSTANT);

#if DEGREES
    phase(c_grad_x1, c_grad_y1, ocv_ref, true);
#elif RADIANS
    phase(c_grad_x1, c_grad_y1, ocv_ref, false);
#endif
    /////////   End Opencv Phase computation API  ///////

    cv::Sobel(in_gray, c_grad_x, ddepth, 1, 0, filter_size, scale, delta, cv::BORDER_CONSTANT);
    cv::Sobel(in_gray, c_grad_y, ddepth, 0, 1, filter_size, scale, delta, cv::BORDER_CONSTANT);

    out_img.create(in_gray.rows, in_gray.cols, CV_16S);

#if DEGREES
    phase(c_grad_x1, c_grad_y1, ocv_ref, true);
#elif RADIANS
    phase(c_grad_x1, c_grad_y1, ocv_ref, false);
#endif
    /////////   End Opencv Phase computation API  ///////

    int rows = in_img.rows;
    int cols = in_img.cols;

    // Call the top function
    phase_accel((ap_uint<INPUT_PTR_WIDTH>*)c_grad_x.data, (ap_uint<INPUT_PTR_WIDTH>*)c_grad_y.data,
                (ap_uint<INPUT_PTR_WIDTH>*)out_img.data, rows, cols);

#if DEGREES
    /////   writing the difference between the OpenCV and the Kernel output into a text file /////
    FILE* fp;
    fp = fopen("diff.txt", "w");
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            short int v = out_img.at<short int>(i, j);
            float v1 = ocv_ref.at<float>(i, j);
            float v2 = v / pow(2.0, 6);
            fprintf(fp, "%f ", v1 - v2); // converting the output fixed point format from Q4.12 format to float
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // Find minimum and maximum differences
    float ocvminvalue, ocvmaxvalue;
    float hlsminvalue, hlsmaxvalue;
    double minval = 65535, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            short int v3 = out_img.at<short int>(i, j);
            float v2 = ocv_ref.at<float>(i, j);
            float v1;

            if (DEGREES) {
                v1 = v3 / (pow(2.0, 6)); // converting the output fixed point format from Q4.12 format to float
            }

            float v = (v2 - v1);

            if (v > 1) cnt++;
            if (minval > v) {
                minval = v;
                ocvminvalue = v2;
                hlsminvalue = v1;
            }
            if (maxval < v) {
                maxval = v;
                ocvmaxvalue = v2;
                hlsmaxvalue = v1;
            }
        }
    }
    printf("Minimum value ocv = %f Minimum value hls = %f\n", ocvminvalue, hlsminvalue);
    printf("Maximum value ocv = %f Maximum value hls = %f\n", ocvmaxvalue, hlsmaxvalue);

#elif RADIANS

    /////   writing the difference between the OpenCV and the Kernel output into a text file /////
    FILE* fp;
    fp = fopen("diff.txt", "w");
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            short int v = out_img.at<short int>(i, j);
            float v1 = ocv_ref.at<float>(i, j);
            float v2 = v / pow(2.0, 12);
            fprintf(fp, "%f ", v1 - v2); // converting the output fixed point format from Q4.12 format to float
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // Find minimum and maximum differences
    float ocvminvalue, ocvmaxvalue;
    float hlsminvalue, hlsmaxvalue;
    double minval = 65535, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            short int v3 = out_img.at<short int>(i, j);
            float v2 = ocv_ref.at<float>(i, j);
            float v1;

            if (RADIANS) {
                v1 = v3 / (pow(2.0, 12)); // converting the output fixed point format from Q4.12 format to float
            }

            float v = (v2 - v1);

            if (v > 1) cnt++;
            if (minval > v) {
                minval = v;
                ocvminvalue = v2;
                hlsminvalue = v1;
            }
            if (maxval < v) {
                maxval = v;
                ocvmaxvalue = v2;
                hlsmaxvalue = v1;
            }
        }
    }
    printf("Minimum value ocv = %f Minimum value hls = %f\n", ocvminvalue, hlsminvalue);
    printf("Maximum value ocv = %f Maximum value hls = %f\n", ocvmaxvalue, hlsmaxvalue);

#endif

    float err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols);
    std::cout << "Minimum error in intensity =" << minval << "\n"
              << "Maximum error in intensity = " << maxval << "\n"
              << "Percentage of pixels above error" << err_per << std::endl;

    in_img.~Mat();
    in_gray.~Mat();
    c_grad_x.~Mat();
    c_grad_y.~Mat();
    c_grad_x1.~Mat();
    c_grad_y1.~Mat();
    ocv_ref.~Mat();
    out_img.~Mat();
    diff.~Mat();

    if (err_per > 0.0f) {
        return 1;
    }

    return 0;
}
