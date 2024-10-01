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

#include "./multitaskv3_imp.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/nnpp/multitaskv3.hpp"

using namespace std;

namespace vitis {
namespace ai {

MultiTaskv3Imp::MultiTaskv3Imp(const std::string& model_name,
                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<MultiTaskv3>(model_name, need_preprocess),
      processor_{vitis::ai::MultiTaskv3PostProcess::create(
          configurable_dpu_task_->getInputTensor(),
          configurable_dpu_task_->getOutputTensor(),
          configurable_dpu_task_->getConfig())} {}
MultiTaskv3Imp::~MultiTaskv3Imp() {}

void MultiTaskv3Imp::run_it(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, cv::INTER_NEAREST);
  } else {
    image = input_image;
  }
  __TIC__(MULTITASKV3_SET_IMG)

  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(MULTITASKV3_SET_IMG)

  __TIC__(MULTITASKV3_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(MULTITASKV3_DPU)
  return;
}

void MultiTaskv3Imp::run_it(const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  for (auto i = 0u; i < input_images.size(); i++) {
    cv::Mat image;
    if (size != input_images[i].size()) {
      cv::resize(input_images[i], image, size, 0, 0, cv::INTER_LINEAR);
      images.push_back(image);
    } else {
      images.push_back(input_images[i]);
    }
  }
  __TIC__(MULTITASKV3_SET_IMG)

  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(MULTITASKV3_SET_IMG)

  __TIC__(MULTITASKV3_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(MULTITASKV3_DPU)
  return;
}

MultiTaskv3Result MultiTaskv3Imp::run_8UC1(const cv::Mat& input_image) {
  run_it(input_image);
  return processor_->post_process(1u)[0];
}

std::vector<MultiTaskv3Result> MultiTaskv3Imp::run_8UC1(
    const std::vector<cv::Mat>& input_images) {
  run_it(input_images);
  return processor_->post_process(input_images.size());
}

MultiTaskv3Result MultiTaskv3Imp::run_8UC3(const cv::Mat& input_image) {
  run_it(input_image);
  return processor_->post_process_visualization(1u)[0];
}

std::vector<MultiTaskv3Result> MultiTaskv3Imp::run_8UC3(
    const std::vector<cv::Mat>& input_images) {
  run_it(input_images);
  return processor_->post_process_visualization(input_images.size());
}

}  // namespace ai
}  // namespace vitis
