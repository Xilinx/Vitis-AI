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
#include <iostream>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/ssd.hpp>
using namespace std;
namespace vitis {
namespace ai {

struct SSDPoseDetect {
  static std::unique_ptr<SSDPoseDetect> create();
  SSDPoseDetect();
  std::vector<vitis::ai::PoseDetectResult> run(const cv::Mat& input_image);
  std::vector<std::vector<vitis::ai::PoseDetectResult>> run(
      const std::vector<cv::Mat>& input_images);
  int getInputWidth();
  int getInputHeight();
  size_t get_input_batch();

 private:
  std::unique_ptr<vitis::ai::SSD> ssd_;
  std::unique_ptr<vitis::ai::PoseDetect> pose_detect_;
};

std::unique_ptr<SSDPoseDetect> SSDPoseDetect::create() {
  return std::unique_ptr<SSDPoseDetect>(new SSDPoseDetect());
}
int SSDPoseDetect::getInputWidth() { return ssd_->getInputWidth(); }
int SSDPoseDetect::getInputHeight() { return ssd_->getInputHeight(); }
size_t SSDPoseDetect::get_input_batch() { return ssd_->get_input_batch(); }

SSDPoseDetect::SSDPoseDetect()
    : ssd_{vitis::ai::SSD::create("ssd_pedestrian_pruned_0_97", true)},
      pose_detect_{vitis::ai::PoseDetect::create("sp_net")} {}

std::vector<vitis::ai::PoseDetectResult> SSDPoseDetect::run(
    const cv::Mat& input_image) {
  std::vector<vitis::ai::PoseDetectResult> mt_results;
  cv::Mat image;
  auto size = cv::Size(ssd_->getInputWidth(), ssd_->getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  // run ssd
  auto ssd_results = ssd_->run(image);

  for (auto& box : ssd_results.bboxes) {
    if (0)
      DLOG(INFO) << "box.x " << box.x << " "            //
                 << "box.y " << box.y << " "            //
                 << "box.width " << box.width << " "    //
                 << "box.height " << box.height << " "  //
                 << "box.score " << box.score << " "    //
          ;
    // int label = box.label;
    int xmin = box.x * input_image.cols;
    int ymin = box.y * input_image.rows;
    int xmax = xmin + box.width * input_image.cols;
    int ymax = ymin + box.height * input_image.rows;
    float confidence = box.score;
    if (confidence < 0.55) continue;
    xmin = std::min(std::max(xmin, 0), input_image.cols);
    xmax = std::min(std::max(xmax, 0), input_image.cols);
    ymin = std::min(std::max(ymin, 0), input_image.rows);
    ymax = std::min(std::max(ymax, 0), input_image.rows);
    cv::Rect roi = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    cv::Mat sub_img = input_image(roi);
    // process each result of ssd detection
    auto single_result = pose_detect_->run(sub_img);
    for (size_t i = 0; i < 28; i = i + 2) {
      ((float*)&single_result.pose14pt)[i] =
          ((float*)&single_result.pose14pt)[i] * sub_img.cols;
      ((float*)&single_result.pose14pt)[i] =
          (((float*)&single_result.pose14pt)[i] + xmin) / input_image.cols;
      ((float*)&single_result.pose14pt)[i + 1] =
          ((float*)&single_result.pose14pt)[i + 1] * sub_img.rows;
      ((float*)&single_result.pose14pt)[i + 1] =
          (((float*)&single_result.pose14pt)[i + 1] + ymin) / input_image.rows;
    }
    mt_results.emplace_back(single_result);
  }
  return mt_results;
}

std::vector<std::vector<vitis::ai::PoseDetectResult>> SSDPoseDetect::run(
    const std::vector<cv::Mat>& input_images) {
  std::vector<std::vector<vitis::ai::PoseDetectResult>> rets;
  for (auto& image : input_images) {
    rets.push_back(run(image));
  }
  return rets;
}

}  // namespace ai
}  // namespace vitis
