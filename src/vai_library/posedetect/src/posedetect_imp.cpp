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
#include "./posedetect_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/globalavepool.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;

namespace vitis {
namespace ai {

PoseDetectImp::PoseDetectImp(const std::string &model_name,
                             bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<PoseDetect>(model_name, need_preprocess) {
}

PoseDetectImp::~PoseDetectImp() {}

PoseDetectResult PoseDetectImp::run(const cv::Mat &input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  __TIC__(POSEDETECT_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(POSEDETECT_SET_IMG)
  __TIC__(POSEDETECT_DPU_CONV)
  configurable_dpu_task_->run(0);
  __TOC__(POSEDETECT_DPU_CONV)
  vitis::ai::globalAvePool(
      (int8_t *)configurable_dpu_task_->getOutputTensor()[0][0].get_data(0),
      184, 4, 7,
      (int8_t *)configurable_dpu_task_->getInputTensor()[1][0].get_data(0), 8);
  __TIC__(POSEDETECT_DPU_FC)
  configurable_dpu_task_->run(1);
  __TOC__(POSEDETECT_DPU_FC)
  auto ret = vitis::ai::pose_detect_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(), image.size());
  return ret[0];
}

std::vector<PoseDetectResult> PoseDetectImp::run(
    const std::vector<cv::Mat> &input_images) {
  std::vector<cv::Mat> images;
  for (auto i = 0u; i < input_images.size(); i++) {
    auto input_image = input_images[i];
    cv::Mat image;
    auto size = cv::Size(getInputWidth(), getInputHeight());
    if (size != input_image.size()) {
      cv::resize(input_image, image, size);
      images.push_back(image);
    } else {
      images.push_back(input_image);
    }
  }
  __TIC__(POSEDETECT_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(POSEDETECT_SET_IMG)
  __TIC__(POSEDETECT_DPU_CONV)
  configurable_dpu_task_->run(0);
  __TOC__(POSEDETECT_DPU_CONV)
  for (auto i = 0u; i < configurable_dpu_task_->getOutputTensor()[0][0].batch;
       i++) {
    vitis::ai::globalAvePool(
        (int8_t *)configurable_dpu_task_->getOutputTensor()[0][0].get_data(i),
        184, 4, 7,
        (int8_t *)configurable_dpu_task_->getInputTensor()[1][0].get_data(i),
        8);
  }
  __TIC__(POSEDETECT_DPU_FC)
  configurable_dpu_task_->run(1);
  __TOC__(POSEDETECT_DPU_FC)

  auto ret = vitis::ai::pose_detect_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(), images[0].size());
  return ret;
}

}  // namespace ai
}  // namespace vitis
