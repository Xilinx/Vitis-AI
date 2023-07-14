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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat process_result(cv::Mat &image,
                       const vitis::ai::FaceLandmarkResult &result,
                       bool is_jpeg) {
  auto points = result.points;

  LOG_IF(INFO, is_jpeg) << "points ";  //
  for (int i = 0; i < 5; ++i) {
    LOG_IF(INFO, is_jpeg) << points[i].first << " " << points[i].second << " ";
    auto point = cv::Point{static_cast<int>(points[i].first * image.cols),
                           static_cast<int>(points[i].second * image.rows)};
    cv::circle(image, point, 3, cv::Scalar(255, 8, 18), -1);
  }
  return image;
}
