/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#pragma once
#include <sys/stat.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vitis/ai/medicalsegmentation.hpp"

using namespace cv;
using namespace std;

Scalar colors[] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
                   Scalar(255, 0, 255), Scalar(0, 255, 255)};
std::vector<std::string> classTypes = {"BE", "cancer", "HGD", "polyp",
                                       "suspicious"};

std::string filenamepart1 = "testfile";

static cv::Mat process_result(
    cv::Mat &m1, const vitis::ai::MedicalSegmentationResult &result,
    bool is_jpeg) {
  cv::Mat img_save;
  cv::Mat img_orig(m1);
  // if dir doesn't exist, create it.
  for (int i = 0; i < 6; i++) {
    std::string path = "results";
    if (i != 0) {
      path = path + "/" + classTypes[i - 1];
    }
    auto ret = mkdir(path.c_str(), 0777);
    if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
      std::cout << "error occured when mkdir " << path << std::endl;
      return cv::Mat();
    }
    if (i == 0) continue;
    std::string fname("results/" + classTypes[i - 1] + "/" + filenamepart1 +
                      ".png");
    cv::resize(result.segmentation[i - 1], img_save, m1.size(), 0, 0,
               cv::INTER_LINEAR);
    cv::imwrite(fname, img_save);          // Save the result as an image;
    auto img_true = cv::imread(fname, 0);  // gray
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours(img_true, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
    if (contours.size()) {
        cv::drawContours(img_orig, contours, -1, colors[i - 1], 3);
        auto midVal = int(contours[0].size()/2);
        cv::putText(img_orig,
                  classTypes[i-1],
                  cv::Point(contours[0][midVal].x, contours[0][midVal].y),
                  cv::FONT_HERSHEY_SIMPLEX,
                  1,
                  cv::Scalar(255,255,255),
                  2,
                  cv::LINE_AA);
      // auto midVal = int(contours[0].size()/2);
      // cv2.putText(raw_img ,classTypes[i],(contours[0][midVal][0][0],
      // contours[0][midVal][0][1]), cv2.FONT_HERSHEY_SIMPLEX,
      // 1,(255,255,255),2,cv2.LINE_AA)
    }
  }
  return img_orig;
}

