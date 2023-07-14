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
#include "./reid_imp.hpp"

#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

ReidImp::ReidImp(const std::string& model_name, bool need_preprocess)
    : Reid(model_name, need_preprocess) {}

ReidImp::~ReidImp() {}

ReidResult ReidImp::run(const cv::Mat& input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  __TIC__(REID_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(REID_SET_IMG)
  __TIC__(REID_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(REID_DPU)
  __TIC__(REID_POST_PROCESS)
  auto ret =
      vitis::ai::reid_post_process(configurable_dpu_task_->getInputTensor(),
                                   configurable_dpu_task_->getOutputTensor(),
                                   configurable_dpu_task_->getConfig(), 0);
  __TOC__(REID_POST_PROCESS)
  return ret;
}

std::vector<ReidResult> ReidImp::run(const std::vector<cv::Mat>& input_images) {
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
  __TIC__(REID_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(REID_SET_IMG)
  __TIC__(REID_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(REID_DPU)
  __TIC__(REID_POST_PROCESS)
  auto ret =
      vitis::ai::reid_post_process(configurable_dpu_task_->getInputTensor(),
                                   configurable_dpu_task_->getOutputTensor(),
                                   configurable_dpu_task_->getConfig());
  __TOC__(REID_POST_PROCESS)
  return ret;
}
std::vector<ReidResult> ReidImp::run(
    const std::vector<vart::xrt_bo_t>& input_bos) {
  __TIC__(REID_DPU)
  configurable_dpu_task_->run_with_xrt_bo(input_bos);
  __TOC__(REID_DPU)
  __TIC__(REID_POST_PROCESS)
  auto ret =
      vitis::ai::reid_post_process(configurable_dpu_task_->getInputTensor(),
                                   configurable_dpu_task_->getOutputTensor(),
                                   configurable_dpu_task_->getConfig());
  __TOC__(REID_POST_PROCESS)
  return ret;
}

}  // namespace ai
}  // namespace vitis
