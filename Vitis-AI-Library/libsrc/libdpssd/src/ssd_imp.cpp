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
#include "./ssd_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xilinx/ai/env_config.hpp>
#include <xilinx/ai/profiling.hpp>

using namespace std;
namespace xilinx {
namespace ai {
DEF_ENV_PARAM(ENABLE_SSD_DEBUG, "0");

SSDImp::SSDImp(const std::string &model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<SSD>(model_name, need_preprocess),
      is_tf{configurable_dpu_task_->getConfig().is_tf()},
      processor_{xilinx::ai::SSDPostProcess::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig())} {}

SSDImp::~SSDImp() {}

SSDResult SSDImp::run(const cv::Mat &input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }
  __TIC__(SSD_total)
  __TIC__(SSD_setimg)
  if (is_tf) {
    configurable_dpu_task_->setInputImageRGB(img);
  } else {
    configurable_dpu_task_->setInputImageBGR(img);
  }

  __TOC__(SSD_setimg)
  __TIC__(SSD_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(SSD_dpu)

  __TIC__(SSD_post)
  auto results = processor_->ssd_post_process();
  __TOC__(SSD_post)

  __TOC__(SSD_total)
  return results;
}

} // namespace ai
} // namespace xilinx
