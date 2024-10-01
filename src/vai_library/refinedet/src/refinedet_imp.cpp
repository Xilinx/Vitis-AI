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
#include "./refinedet_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;
namespace vitis {
namespace ai {

RefineDetImp::RefineDetImp(const std::string& model_name, bool need_preprocess)
    : RefineDet(model_name, need_preprocess),
      is_tf_(configurable_dpu_task_->getConfig().is_tf()) {
  if (is_tf_) {
    tfprocessor_ = vitis::ai::TFRefineDetPostProcess::create(
        configurable_dpu_task_->getInputTensor()[0],
        configurable_dpu_task_->getOutputTensor()[0],
        configurable_dpu_task_->getConfig());
  } else {
    processor_ = vitis::ai::RefineDetPostProcess::create(
        configurable_dpu_task_->getInputTensor()[0],
        configurable_dpu_task_->getOutputTensor()[0],
        configurable_dpu_task_->getConfig());
  }
}

RefineDetImp::~RefineDetImp() {}

RefineDetResult RefineDetImp::run(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0);
  } else {
    image = input_image;
  }
  __TIC__(DPREFINEDET_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(DPREFINEDET_SET_IMG)

  __TIC__(DPREFINEDET_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(DPREFINEDET_DPU)
  __TIC__(DPREFINEDET_POST_ARM)
  RefineDetResult results;
  if (is_tf_) {
    results = tfprocessor_->tfrefinedet_post_process(1u)[0];
  } else {
    results = processor_->refine_det_post_process(1u)[0];
  }
  __TOC__(DPREFINEDET_POST_ARM)
  return results;
}

std::vector<RefineDetResult> RefineDetImp::run(
    const std::vector<cv::Mat>& input_images) {
  vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  for (auto& input_image : input_images) {
    cv::Mat img;
    if (size != input_image.size()) {
      cv::resize(input_image, img, size);
    } else {
      img = input_image;
    }
    images.push_back(img);
  }
  __TIC__(DPREFINEDET_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(DPREFINEDET_SET_IMG)

  __TIC__(DPREFINEDET_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(DPREFINEDET_DPU)
  __TIC__(DPREFINEDET_POST_ARM)
  std::vector<RefineDetResult> results;
  if (is_tf_) {
    results = tfprocessor_->tfrefinedet_post_process(images.size());
  } else {
    results = processor_->refine_det_post_process(images.size());
  }
  __TOC__(DPREFINEDET_POST_ARM)
  return results;
}

std::vector<RefineDetResult> RefineDetImp::run(
    const std::vector<vart::xrt_bo_t>& input_bos) {
  __TIC__(DPREFINEDET_DPU)
  configurable_dpu_task_->run_with_xrt_bo(input_bos);
  __TOC__(DPREFINEDET_DPU)

  __TIC__(DPREFINEDET_POST_ARM)
  std::vector<RefineDetResult> results;
  if (is_tf_) {
    results = tfprocessor_->tfrefinedet_post_process(input_bos.size());
  } else {
    results = processor_->refine_det_post_process(input_bos.size());
  }
  __TOC__(DPREFINEDET_POST_ARM)
  return results;
}

}  // namespace ai
}  // namespace vitis
