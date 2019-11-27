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
#define ENABLE_NEON
#include "./refinedet_imp.hpp"
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xilinx/ai/env_config.hpp>
#include <xilinx/ai/profiling.hpp>
using namespace std;
namespace xilinx {
namespace ai {

RefineDetImp::RefineDetImp(const std::string& model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<RefineDet>(model_name, need_preprocess),
    processor_{xilinx::ai::RefineDetPostProcess::create(
        configurable_dpu_task_->getInputTensor(), configurable_dpu_task_->getOutputTensor(),
        configurable_dpu_task_->getConfig())} {
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
  auto results = processor_->refine_det_post_process();
  __TOC__(DPREFINEDET_POST_ARM)
  return results;
}

}
}
