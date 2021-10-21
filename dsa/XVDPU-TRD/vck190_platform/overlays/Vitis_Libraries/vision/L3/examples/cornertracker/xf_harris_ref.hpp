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
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "xf_harris_ref_def.hpp"

void harris(cv::Mat in_img, std::vector<cv::Point2f>& ocv_points) {
    float Th;
    if (FILTER_WIDTH == 3)
        Th = 30532960.00;
    else if (FILTER_WIDTH == 5)
        Th = 902753878016.0;
    else if (FILTER_WIDTH == 7)
        Th = 41151168289701888.000000;

    cv::Mat ocv_out_img;
    cv::Mat ocvpnts;
    ocv_out_img.create(in_img.rows, in_img.cols, CV_8U); // create memory for opencv output image

    harrisOCV::ocv_ref(in_img, ocv_out_img, Th * 40);
    // 	ocvpnts = in_img.clone();

    // Drawing a circle around corners
    for (int j = 0; j < in_img.rows; j++) {
        for (int i = 0; i < in_img.cols; i++) {
            if ((int)ocv_out_img.at<unsigned char>(j, i)) {
                // 				cv::circle(ocvpnts, cv::Point(i, j), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
                ocv_points.push_back(cv::Point2f(i, j));
            }
        }
    }

    printf("ocv corner count = %d\n", ocv_points.size());
    return;
}
