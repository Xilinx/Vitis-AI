/*
 * Copyright 2019 Xilinx Inc.
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
#include <xilinx/ai/posedetect.hpp>
#include <xilinx/ai/ssd.hpp>

namespace xilinx {
namespace ai {

struct SSDPoseDetect {
  static std::unique_ptr<SSDPoseDetect> create();
  SSDPoseDetect();
  std::vector<xilinx::ai::PoseDetectResult> run(const cv::Mat &input_image);
  int getInputWidth();
  int getInputHeight();

private:
  std::unique_ptr<xilinx::ai::SSD> ssd_;
  std::unique_ptr<xilinx::ai::PoseDetect> pose_detect_;
};

std::unique_ptr<SSDPoseDetect> SSDPoseDetect::create() {
  return std::unique_ptr<SSDPoseDetect>(new SSDPoseDetect());
}
int SSDPoseDetect::getInputWidth() { return ssd_->getInputWidth(); }
int SSDPoseDetect::getInputHeight() { return ssd_->getInputHeight(); }

SSDPoseDetect::SSDPoseDetect()
    : ssd_{xilinx::ai::SSD::create("ssd_pedestrain_pruned_0_97",
                                   true)},
      pose_detect_{xilinx::ai::PoseDetect::create("sp_net")} {}

std::vector<xilinx::ai::PoseDetectResult>
SSDPoseDetect::run(const cv::Mat &input_image) {
  std::vector<xilinx::ai::PoseDetectResult> mt_results;
  cv::Mat image;
  auto size = cv::Size(ssd_->getInputWidth(), ssd_->getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  auto ssd_results = ssd_->run(image);

  for (auto &box : ssd_results.bboxes) {

    int xmin = box.x * input_image.cols;
    int ymin = box.y * input_image.rows;
    int xmax = xmin + box.width * input_image.cols;
    int ymax = ymin + box.height * input_image.rows;
    float confidence = box.score;
    if (confidence < 0.55)
      continue;
    xmin = std::min(std::max(xmin, 0), input_image.cols);
    xmax = std::min(std::max(xmax, 0), input_image.cols);
    ymin = std::min(std::max(ymin, 0), input_image.rows);
    ymax = std::min(std::max(ymax, 0), input_image.rows);
    cv::Rect roi = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    cv::Mat sub_img = input_image(roi);

    auto single_result = pose_detect_->run(sub_img);
    for (size_t i = 0; i < 28; i = i + 2) {
      ((float *)&single_result.pose14pt)[i] =
          ((float *)&single_result.pose14pt)[i] * sub_img.cols;
      ((float *)&single_result.pose14pt)[i] =
          (((float *)&single_result.pose14pt)[i] + xmin) / input_image.cols;
      ((float *)&single_result.pose14pt)[i + 1] =
          ((float *)&single_result.pose14pt)[i + 1] * sub_img.rows;
      ((float *)&single_result.pose14pt)[i + 1] =
          (((float *)&single_result.pose14pt)[i + 1] + ymin) / input_image.rows;
    }
    mt_results.push_back(single_result);
  }
  return mt_results;
}
} // namespace ai
} // namespace xilinx
