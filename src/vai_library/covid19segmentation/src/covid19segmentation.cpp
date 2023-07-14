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
#include <vitis/ai/covid19segmentation.hpp>

#include "./covid19segmentation_imp.hpp"

namespace vitis {
namespace ai {

Covid19Segmentation::Covid19Segmentation() {}
Covid19Segmentation::~Covid19Segmentation() {}

std::unique_ptr<Covid19Segmentation> Covid19Segmentation::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<Covid19Segmentation>(
      new Covid19SegmentationImp(model_name, need_preprocess));
}

// begin Covid19Segmentation8UC1 implementaton......

std::unique_ptr<Covid19Segmentation8UC1> Covid19Segmentation8UC1::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<Covid19Segmentation8UC1>(
      new Covid19Segmentation8UC1(Covid19Segmentation::create(model_name, need_preprocess)));
}

Covid19Segmentation8UC1::Covid19Segmentation8UC1(std::unique_ptr<Covid19Segmentation> covid19segmentation)
    : covid19segmentation_{std::move(covid19segmentation)} {}

Covid19Segmentation8UC1::~Covid19Segmentation8UC1() {}

int Covid19Segmentation8UC1::getInputWidth() const {
  return covid19segmentation_->getInputWidth();
}

int Covid19Segmentation8UC1::getInputHeight() const {
  return covid19segmentation_->getInputHeight();
}

size_t Covid19Segmentation8UC1::get_input_batch() const {
  return covid19segmentation_->get_input_batch();
}

Covid19SegmentationResult Covid19Segmentation8UC1::run(const cv::Mat& image) {
  return covid19segmentation_->run_8UC1(image);
}

std::vector<Covid19SegmentationResult> Covid19Segmentation8UC1::run(
    const std::vector<cv::Mat>& images) {
  return covid19segmentation_->run_8UC1(images);
}

std::unique_ptr<Covid19Segmentation8UC3> Covid19Segmentation8UC3::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<Covid19Segmentation8UC3>(
      new Covid19Segmentation8UC3(Covid19Segmentation::create(model_name, need_preprocess)));
}

Covid19Segmentation8UC3::Covid19Segmentation8UC3(std::unique_ptr<Covid19Segmentation> covid19segmentation)
    : covid19segmentation_{std::move(covid19segmentation)} {}

Covid19Segmentation8UC3::~Covid19Segmentation8UC3() {}

int Covid19Segmentation8UC3::getInputWidth() const {
  return covid19segmentation_->getInputWidth();
}

int Covid19Segmentation8UC3::getInputHeight() const {
  return covid19segmentation_->getInputHeight();
}

size_t Covid19Segmentation8UC3::get_input_batch() const {
  return covid19segmentation_->get_input_batch();
}

Covid19SegmentationResult Covid19Segmentation8UC3::run(const cv::Mat& image) {
  return covid19segmentation_->run_8UC3(image);
}

std::vector<Covid19SegmentationResult> Covid19Segmentation8UC3::run(
    const std::vector<cv::Mat>& images) {
  return covid19segmentation_->run_8UC3(images);
}

}  // namespace ai
}  // namespace vitis
