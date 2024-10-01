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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Scalar label2Color(int label) {
  static cv::Scalar table[19] = {
      {128, 64, 128},  {244, 35, 232},  {70, 70, 70},   {102, 102, 156},
      {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0},
      {107, 142, 35},  {152, 251, 152}, {0, 130, 180},  {220, 20, 60},
      {255, 0, 0},     {0, 0, 142},     {0, 0, 70},     {0, 60, 100},
      {0, 80, 100},    {0, 0, 230},     {119, 11, 32}};
  return table[label % 19];
}

static cv::Mat process_result(cv::Mat &image,
                              const vitis::ai::YOLOv2Result &result,
                              bool is_jpeg) {
  for (const auto& bbox : result.bboxes) {
    int label = bbox.label;
    float xmin = bbox.x * image.cols + 1;
    float ymin = bbox.y * image.rows + 1;
    float xmax = xmin + bbox.width * image.cols;
    float ymax = ymin + bbox.height * image.rows;
    float confidence = bbox.score;
    if (xmax > image.cols) xmax = image.cols;
    if (ymax > image.rows) ymax = image.rows;
    LOG_IF(INFO, is_jpeg) << "RESULT: " << label << "\t" << xmin << "\t" << ymin
                          << "\t" << xmax << "\t" << ymax << "\t" << confidence
                          << "\n";
    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  label2Color(label), 2, 1, 0);
  }
  return image;
}
