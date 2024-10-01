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
#include "./tfssd_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/graph/graph.hpp>

using namespace std;
namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_SSD_DEBUG, "0");

TFSSDImp::TFSSDImp(const std::string& model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<TFSSD>(model_name, need_preprocess),
      processor_{vitis::ai::TFSSDPostProcess::create(
          model_name,
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig(),
          (configurable_dpu_task_->get_graph())
              ->get_attr<std::string>("dirname"),
          real_batch_size )} {}

TFSSDImp::~TFSSDImp() {}

TFSSDResult TFSSDImp::run(const cv::Mat& input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }
  __TIC__(SSD_total)
  __TIC__(SSD_setimg)
  real_batch_size = 1;
  configurable_dpu_task_->setInputImageRGB(img);

  __TOC__(SSD_setimg)
  __TIC__(SSD_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(SSD_dpu)

  __TIC__(SSD_post)
  auto results = processor_->ssd_post_process();
  __TOC__(SSD_post)

  __TOC__(SSD_total)
  return results[0];
}

std::vector<TFSSDResult> TFSSDImp::run(const std::vector<cv::Mat>& input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());
  auto batch_size = get_input_batch();
  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  std::vector<cv::Mat> vimg(real_batch_size);
  for (auto i = 0; i < real_batch_size; i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);
    } else {
      vimg[i] = input_img[i];
    }
  }
  __TIC__(SSD_total)
  __TIC__(SSD_setimg)

  configurable_dpu_task_->setInputImageRGB(vimg);

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

}  // namespace ai
}  // namespace vitis
