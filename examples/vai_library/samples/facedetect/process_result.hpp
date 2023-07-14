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

cv::Mat process_result(cv::Mat &m1, const vitis::ai::FaceDetectResult &result,
                       bool is_jpeg) {
  cv::Mat image;
  cv::resize(m1, image, cv::Size{result.width, result.height});
  for (const auto &r : result.rects) {
    LOG_IF(INFO, is_jpeg) << " " << r.score << " "  //
                          << r.x << " "             //
                          << r.y << " "             //
                          << r.width << " "         //
                          << r.height;
    cv::rectangle(image,
                  cv::Rect{cv::Point(r.x * image.cols, r.y * image.rows),
                           cv::Size{(int)(r.width * image.cols),
                                    (int)(r.height * image.rows)}},
                  0xff);
  }

  return image;
}
