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

#include "./vehicleclassification_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <vector>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

VehicleClassificationImp::VehicleClassificationImp(
    const std::string& model_name, bool need_preprocess)
    : VehicleClassification(model_name, need_preprocess) {}

VehicleClassificationImp::~VehicleClassificationImp() {}

vitis::ai::VehicleClassificationResult VehicleClassificationImp::run(
    const cv::Mat& input_image) {
  cv::Mat image;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  if (size == input_image.size()) {
    image = input_image;
  } else {
    cv::resize(input_image, image, size);
  }
  //__TIC__(CLASSIFY_E2E_TIME)
  __TIC__(CLASSIFY_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(CLASSIFY_SET_IMG)

  auto postprocess_index = 0;
  __TIC__(CLASSIFY_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(CLASSIFY_DPU)
  __TIC__(CLASSIFY_POST_ARM)
  auto ret = vehicleclassification_post_process(
      configurable_dpu_task_->getInputTensor()[postprocess_index],
      configurable_dpu_task_->getOutputTensor()[postprocess_index],
      configurable_dpu_task_->getConfig());

  ret[0].type = 1;
  if (!configurable_dpu_task_->getConfig()
           .vehicleclassification_param()
           .label_type()
           .empty()) {
    auto label_type = configurable_dpu_task_->getConfig()
                          .vehicleclassification_param()
                          .label_type();
    if (label_type == "VEHICLE_MAKE") {
      ret[0].type = 1;
    } else if (label_type == "VEHICLE_TYPE") {
      ret[0].type = 2;
    }
  }

  __TOC__(CLASSIFY_POST_ARM)
  //__TOC__(CLASSIFY_E2E_TIME)

  return ret[0];
}

std::vector<VehicleClassificationResult> VehicleClassificationImp::run(
    const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);

  for (auto i = 0u; i < input_images.size(); i++) {
    if (size == input_images[i].size()) {
      images.push_back(input_images[i]);
    } else {
      cv::Mat image;
      cv::resize(input_images[i], image, size);
      images.push_back(image);
    }
  }

  __TIC__(CLASSIFY_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(CLASSIFY_SET_IMG)

  auto postprocess_index = 0;
  __TIC__(CLASSIFY_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(CLASSIFY_DPU)

  __TIC__(CLASSIFY_POST_ARM)
  auto rets = vehicleclassification_post_process(
      configurable_dpu_task_->getInputTensor()[postprocess_index],
      configurable_dpu_task_->getOutputTensor()[postprocess_index],
      configurable_dpu_task_->getConfig());

  for (auto& ret : rets) {
    ret.type = 1;
    if (!configurable_dpu_task_->getConfig()
             .vehicleclassification_param()
             .label_type()
             .empty()) {
      auto label_type = configurable_dpu_task_->getConfig()
                            .vehicleclassification_param()
                            .label_type();
      if (label_type == "VEHICLE_MAKE") {
        ret.type = 1;
      } else if (label_type == "VEHICLE_TYPE") {
        ret.type = 2;
      }
    }
  }
  __TOC__(CLASSIFY_POST_ARM)
  //__TOC__(CLASSIFY_E2E_TIME)
  return rets;
}

}  // namespace ai
}  // namespace vitis
