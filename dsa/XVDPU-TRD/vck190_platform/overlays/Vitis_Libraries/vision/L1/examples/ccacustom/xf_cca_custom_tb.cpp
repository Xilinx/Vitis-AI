/*
 * Copyright 2020 Xilinx, Inc.
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
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"

#include "xf_cca_custom_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, out_img;

    // reading in the image
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    cv::Mat threshold_binary_image, median_filter_image1, post_blur_threshold_binary_image;
    cv::threshold(in_img, threshold_binary_image, 60, 255, cv::THRESH_BINARY_INV);
    cv::medianBlur(threshold_binary_image, median_filter_image1, 9);
    cv::threshold(median_filter_image1, post_blur_threshold_binary_image, 10, 255, cv::THRESH_BINARY);
    cv::imwrite("in_thresh.png", post_blur_threshold_binary_image);

    out_img.create(in_img.rows, in_img.cols, in_img.type());
    int height = in_img.rows;
    int width = in_img.cols;
    unsigned char* tmp_out_data1 = (unsigned char*)malloc(height * width);
    unsigned char* tmp_out_data2 = (unsigned char*)malloc(height * width);
    int def_pix, obj_pix;

    ////////////////////	HLS TOP function call	/////////////////
    cca_custom_accel((uint8_t*)post_blur_threshold_binary_image.data, (uint8_t*)post_blur_threshold_binary_image.data,
                     (uint8_t*)tmp_out_data1, (uint8_t*)tmp_out_data2, (uint8_t*)out_img.data, (int*)&obj_pix,
                     (int*)&def_pix, height, width);

    cv::imwrite("Defect_image.png", out_img);

    cv::Mat median_filter_image2;
    cv::medianBlur(out_img, median_filter_image2, 9);
    cv::imwrite("post_median.png", median_filter_image2);

    double defect_threshold = 0.3; //.3%

    double total_pixels = height * width;
    double obj_wo_def = obj_pix;
    double obj_w_def = obj_pix + def_pix;
    double defect_pixels = def_pix;
    double defect_pixels_post_blur = cv::countNonZero(median_filter_image2);
    double defect_density = (defect_pixels_post_blur / obj_w_def) * 100;
    bool defect_decision = (defect_density > defect_threshold);

    char text_buffer[60];
    int y_point = 100;
    cv::Mat text_image = cv::Mat::zeros(height, width, CV_8U);

    sprintf(text_buffer, "Total Pixels: %.2lf", total_pixels);
    cv::putText(text_image, text_buffer, cv::Point(100, y_point), cv::FONT_HERSHEY_TRIPLEX, 1, 255, 1);
    y_point += 30;

    sprintf(text_buffer, "Total Object Pixels: %.2lf", obj_w_def);
    cv::putText(text_image, text_buffer, cv::Point(100, y_point), cv::FONT_HERSHEY_TRIPLEX, 1, 255, 1);
    y_point += 30;

    sprintf(text_buffer, "Defect Pixels: %.2lf", defect_pixels_post_blur);
    cv::putText(text_image, text_buffer, cv::Point(100, y_point), cv::FONT_HERSHEY_TRIPLEX, 1, 255, 1);
    y_point += 30;

    sprintf(text_buffer, "Density Trashold: %.2lf", defect_threshold);
    cv::putText(text_image, text_buffer, cv::Point(100, y_point), cv::FONT_HERSHEY_TRIPLEX, 1, 255, 1);
    y_point += 30;

    sprintf(text_buffer, "Defect Density: %.2lf ", defect_density);
    cv::putText(text_image, text_buffer, cv::Point(100, y_point), cv::FONT_HERSHEY_TRIPLEX, 1, 255, 1);
    y_point += 30;

    sprintf(text_buffer, "Is Defected: %s", defect_decision ? "Yes" : "No");
    cv::putText(text_image, text_buffer, cv::Point(100, y_point), cv::FONT_HERSHEY_TRIPLEX, 1, 255, 1);

    cv::String text_image_window = "./output_8_text_overlay.jpg";
    cv::imwrite(text_image_window, text_image);

    printf("Test passed!\n");

    return 0;
}
