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
#include "./rcan_imp.hpp"

#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

RcanImp::RcanImp(const std::string& model_name, bool need_preprocess)
    : Rcan(model_name, need_preprocess) {}

RcanImp::~RcanImp() {}

RcanResult RcanImp::run(const cv::Mat& input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  __TIC__(RCAN_SET_IMG)
  if (configurable_dpu_task_->getConfig().order_type() == 1) {
    configurable_dpu_task_->setInputImageBGR(image);
  } else if (configurable_dpu_task_->getConfig().order_type() == 2) {
    configurable_dpu_task_->setInputImageRGB(image);
  } else {
    LOG(FATAL) << "unknown image order type";
  }
  __TOC__(RCAN_SET_IMG)
  __TIC__(RCAN_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(RCAN_DPU)
  __TIC__(RCAN_POST_PROCESS)
  auto ret = vitis::ai::rcan_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(), 0, configurable_dpu_task_->getConfig());
  //LOG(INFO) << vitis::ai::library::tensor_scale(configurable_dpu_task_->getOutputTensor()[0][0]);
  //LOG(INFO) << vitis::ai::library::tensor_scale(configurable_dpu_task_->getInputTensor()[0][0]);
  __TOC__(RCAN_POST_PROCESS)
  return ret;
}

std::vector<RcanResult> RcanImp::run(const std::vector<cv::Mat>& input_images) {
  vector<cv::Mat> images;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  for (auto& input_image : input_images) {
    Mat image;
    if (size != input_image.size()) {
      cv::resize(input_image, image, size);
    } else {
      image = input_image;
    }
    images.push_back(image);
  }
  __TIC__(RCAN_SET_IMG)
    //LOG(INFO) << configurable_dpu_task_->getConfig().order_type();
  if (configurable_dpu_task_->getConfig().order_type() == 1) {
    configurable_dpu_task_->setInputImageBGR(images);
  } else if (configurable_dpu_task_->getConfig().order_type() == 2) {
    //LOG(INFO) << "rgb";
    configurable_dpu_task_->setInputImageRGB(images);
  } else {
    LOG(FATAL) << "unknown image order type";
  }
  __TOC__(RCAN_SET_IMG)
  __TIC__(RCAN_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(RCAN_DPU)
  __TIC__(RCAN_POST_PROCESS)
  auto ret =
      vitis::ai::rcan_post_process(configurable_dpu_task_->getInputTensor(),
                                   configurable_dpu_task_->getOutputTensor(), configurable_dpu_task_->getConfig());
  //LOG(INFO) << vitis::ai::library::tensor_scale(configurable_dpu_task_->getOutputTensor()[0][0]);
  __TOC__(RCAN_POST_PROCESS)
  return ret;
}
std::vector<RcanResult> RcanImp::run(
    const std::vector<vart::xrt_bo_t>& input_bos) {
  __TIC__(RCAN_DPU)
  configurable_dpu_task_->run_with_xrt_bo(input_bos);
  __TOC__(RCAN_DPU)
  __TIC__(RCAN_POST_PROCESS)
  auto ret =
      vitis::ai::rcan_post_process(configurable_dpu_task_->getInputTensor(),
                                   configurable_dpu_task_->getOutputTensor(), configurable_dpu_task_->getConfig());
  __TOC__(RCAN_POST_PROCESS)
  return ret;
}

}  // namespace ai
}  // namespace vitis
