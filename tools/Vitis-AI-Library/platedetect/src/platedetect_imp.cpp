/*
 * Copyright 2019 xilinx Inc.
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
#include "./platedetect_imp.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/env_config.hpp"
// #include <fstream>
using namespace std;
namespace vitis {
namespace ai {
int GLOBAL_ENABLE_TEST_ACC = 0;

PlateDetectImp::PlateDetectImp(const std::string &model_name,
                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<PlateDetect>(model_name,
                                                   need_preprocess) {}

PlateDetectImp::PlateDetectImp(const std::string &model_name,
                               xir::Attrs *attrs,
                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<PlateDetect>(model_name,
                                                   attrs,
                                                   need_preprocess) {}

PlateDetectImp::~PlateDetectImp() {}

PlateDetectResult PlateDetectImp::run(const cv::Mat &img) {
  cv::Mat image;
  auto size = cv::Size(configurable_dpu_task_->getInputTensor()[0][0].width,
                       configurable_dpu_task_->getInputTensor()[0][0].height);
  if (size != img.size()) {
    cv::resize(img, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = img;
  }
  __TIC__(PLATEDETECT_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(PLATEDETECT_SET_IMG)

  __TIC__(PLATEDETECT_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(PLATEDETECT_DPU)

  return plate_detect_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor())[0];
}

std::vector<PlateDetectResult> PlateDetectImp::run(
    const std::vector<cv::Mat> &imgs) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  for (auto i = 0u; i < imgs.size(); i++) {
    if (size != imgs[i].size()) {
      cv::Mat img;
      cv::resize(imgs[i], img, size, 0);
      images.push_back(img);
    } else {
      images.push_back(imgs[i]);
    }
  }
  __TIC__(PLATEDETECT_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(PLATEDETECT_SET_IMG)

  __TIC__(PLATEDETECT_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(PLATEDETECT_DPU)
  return plate_detect_post_process(configurable_dpu_task_->getInputTensor(),
                                   configurable_dpu_task_->getOutputTensor());
}

}  // namespace ai
}  // namespace vitis
