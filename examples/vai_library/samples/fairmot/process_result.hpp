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

cv::Mat process_result(cv::Mat& image, const vitis::ai::FairMotResult& result,
                       bool is_jpeg) {
  auto feats = result.feats;
  auto bboxes = result.bboxes;
  auto img = image.clone();
  for (auto i = 0u; i < bboxes.size(); ++i) {
    auto box = bboxes[i];
    float x = box.x * (img.cols);
    float y = box.y * (img.rows);
    int xmin = x;
    int ymin = y;
    int xmax = x + (box.width) * (img.cols);
    int ymax = y + (box.height) * (img.rows);
    float score = box.score;
    xmin = std::min(std::max(xmin, 0), img.cols);
    xmax = std::min(std::max(xmax, 0), img.cols);
    ymin = std::min(std::max(ymin, 0), img.rows);
    ymax = std::min(std::max(ymax, 0), img.rows);

    LOG(INFO) << "RESULT " << box.label << " :\t" << xmin << "\t" << ymin
              << "\t" << xmax << "\t" << ymax << "\t" << score << "\n";
    LOG(INFO) << "feat size: " << feats[i].size()
              << " First 5 digits: " << feats[i].data[0] + 0.0f << " "
              << feats[i].data[1] + 0.0f << " " << feats[i].data[2] + 0.0f
              << " " << feats[i].data[3] + 0.0f << " "
              << feats[i].data[4] + 0.0f << "\n";
    cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  cv::Scalar(0, 255, 0), 1, 1, 0);
  }
  return img;
}
