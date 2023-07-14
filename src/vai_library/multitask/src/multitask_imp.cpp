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

#include "./multitask_imp.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/nnpp/multitask.hpp"

using namespace std;

namespace vitis {
namespace ai {

MultiTaskImp::MultiTaskImp(const std::string& model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<MultiTask>(model_name, need_preprocess),
      processor_{vitis::ai::MultiTaskPostProcess::create(
          configurable_dpu_task_->getInputTensor(),
          configurable_dpu_task_->getOutputTensor(),
          configurable_dpu_task_->getConfig())} {}
MultiTaskImp::~MultiTaskImp() {}

void MultiTaskImp::run_it(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, cv::INTER_NEAREST);
  } else {
    image = input_image;
  }
  __TIC__(MULTITASK_SET_IMG)

  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(MULTITASK_SET_IMG)

  __TIC__(MULTITASK_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(MULTITASK_DPU)
  return;
}

void MultiTaskImp::run_it(const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  for (auto i = 0u; i < std::min(input_images.size(),
                                 configurable_dpu_task_->get_input_batch());
       i++) {
    cv::Mat image;
    if (size != input_images[i].size()) {
      cv::resize(input_images[i], image, size, 0, 0, cv::INTER_LINEAR);
      images.push_back(image);
    } else {
      images.push_back(input_images[i]);
    }
  }
  __TIC__(MULTITASK_SET_IMG)

  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(MULTITASK_SET_IMG)

  __TIC__(MULTITASK_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(MULTITASK_DPU)
  return;
}

MultiTaskResult MultiTaskImp::run_8UC1(const cv::Mat& input_image) {
  run_it(input_image);
  return processor_->post_process_seg(1u)[0];
}

std::vector<MultiTaskResult> MultiTaskImp::run_8UC1(
    const std::vector<cv::Mat>& input_images) {
  run_it(input_images);
  return processor_->post_process_seg(input_images.size());
}

MultiTaskResult MultiTaskImp::run_8UC3(const cv::Mat& input_image) {
  run_it(input_image);
  return processor_->post_process_seg_visualization(1u)[0];
}

std::vector<MultiTaskResult> MultiTaskImp::run_8UC3(
    const std::vector<cv::Mat>& input_images) {
  run_it(input_images);
  return processor_->post_process_seg_visualization(input_images.size());
}

}  // namespace ai
}  // namespace vitis
