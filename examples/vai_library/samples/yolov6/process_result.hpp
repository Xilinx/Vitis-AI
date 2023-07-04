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
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
static cv::Scalar getColor(int label) {
  return cv::Scalar(label * 2, 255 - label * 2, label + 50);
}

static cv::Mat process_result(cv::Mat& image,
                              const vitis::ai::YOLOv6Result& result_in,
                              bool is_jpeg) {
  for (const auto& result : result_in.bboxes) {
    int label = result.label;
    auto& box = result.box;
    LOG_IF(INFO, is_jpeg) << "RESULT: " << label << "\t" << std::fixed
                          << std::setprecision(2) << box[0] << "\t" << box[1]
                          << "\t" << box[2] << "\t" << box[3] << "\t"
                          << std::setprecision(6) << result.score << "\n";

    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
                  getColor(label), 1, 1, 0);
  }
  return image;
}
