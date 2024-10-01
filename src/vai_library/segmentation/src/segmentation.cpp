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
#include <vitis/ai/segmentation.hpp>

#include "./segmentation_imp.hpp"

namespace vitis {
namespace ai {

Segmentation::Segmentation() {}
Segmentation::~Segmentation() {}

std::unique_ptr<Segmentation> Segmentation::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<Segmentation>(
      new SegmentationImp(model_name, need_preprocess));
}

// begin Segmentation8UC1 implementaton......

std::unique_ptr<Segmentation8UC1> Segmentation8UC1::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<Segmentation8UC1>(
      new Segmentation8UC1(Segmentation::create(model_name, need_preprocess)));
}

Segmentation8UC1::Segmentation8UC1(std::unique_ptr<Segmentation> segmentation)
    : segmentation_{std::move(segmentation)} {}

Segmentation8UC1::~Segmentation8UC1() {}

int Segmentation8UC1::getInputWidth() const {
  return segmentation_->getInputWidth();
}

int Segmentation8UC1::getInputHeight() const {
  return segmentation_->getInputHeight();
}

size_t Segmentation8UC1::get_input_batch() const {
  return segmentation_->get_input_batch();
}

SegmentationResult Segmentation8UC1::run(const cv::Mat& image) {
  return segmentation_->run_8UC1(image);
}

std::vector<SegmentationResult> Segmentation8UC1::run(
    const std::vector<cv::Mat>& images) {
  return segmentation_->run_8UC1(images);
}

std::unique_ptr<Segmentation8UC3> Segmentation8UC3::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<Segmentation8UC3>(
      new Segmentation8UC3(Segmentation::create(model_name, need_preprocess)));
}

Segmentation8UC3::Segmentation8UC3(std::unique_ptr<Segmentation> segmentation)
    : segmentation_{std::move(segmentation)} {}

Segmentation8UC3::~Segmentation8UC3() {}

int Segmentation8UC3::getInputWidth() const {
  return segmentation_->getInputWidth();
}

int Segmentation8UC3::getInputHeight() const {
  return segmentation_->getInputHeight();
}

size_t Segmentation8UC3::get_input_batch() const {
  return segmentation_->get_input_batch();
}

SegmentationResult Segmentation8UC3::run(const cv::Mat& image) {
  return segmentation_->run_8UC3(image);
}

std::vector<SegmentationResult> Segmentation8UC3::run(
    const std::vector<cv::Mat>& images) {
  return segmentation_->run_8UC3(images);
}

}  // namespace ai
}  // namespace vitis
