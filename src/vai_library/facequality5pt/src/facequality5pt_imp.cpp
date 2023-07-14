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

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "facequality5pt_imp.hpp"

using std::vector;
using std::string;

namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_FACE_QUALITY5PT_DEBUG, "0"); 

FaceQuality5ptImp::FaceQuality5ptImp(const string& model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<FaceQuality5pt>(model_name, need_preprocess),
      mode_{FaceQuality5pt::Mode::DAY} {
} 

FaceQuality5ptImp::~FaceQuality5ptImp() {}

FaceQuality5pt::Mode FaceQuality5ptImp::getMode() { return mode_; }
void FaceQuality5ptImp::setMode(FaceQuality5pt::Mode mode) { mode_ = mode; }

FaceQuality5ptResult FaceQuality5ptImp::run(const cv::Mat& input_image) {
  __TIC__(FACE_QUALITY5PT_E2E)
  cv::Mat image;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  if (size != input_image.size()) {
      cv::resize(input_image, image, size, 0);
  } else {
    image = input_image;
  }
  __TIC__(FACE_QUALITY5PT_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(FACE_QUALITY5PT_SET_IMG)

  __TIC__(FACE_QUALITY5PT_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FACE_QUALITY5PT_DPU)
  
  if (mode_ == FaceQuality5pt::Mode::NIGHT) {
     auto ret = vitis::ai::face_quality5pt_post_process(
            configurable_dpu_task_->getInputTensor(),
            configurable_dpu_task_->getOutputTensor(),
            configurable_dpu_task_->getConfig(), false);
  __TOC__(FACE_QUALITY5PT_E2E)
     return ret[0];

  } else {
    auto ret = vitis::ai::face_quality5pt_post_process(
           configurable_dpu_task_->getInputTensor(),
           configurable_dpu_task_->getOutputTensor(),
           configurable_dpu_task_->getConfig(), true);
  __TOC__(FACE_QUALITY5PT_E2E)
    return ret[0];
  }
}

//FaceQuality5ptResult FaceQuality5ptImp::run_original(const cv::Mat& input_image) {
//  __TIC__(FACE_QUALITY5PT_E2E)
//  cv::Mat image;
//  int width = getInputWidth();
//  int height = getInputHeight();
//  auto size = cv::Size(width, height);
//  if (size != input_image.size()) {
//      cv::resize(input_image, image, size, 0);
//  } else {
//    image = input_image;
//  }
//  __TIC__(FACE_QUALITY5PT_SET_IMG)
//  configurable_dpu_task_->setInputImageBGR(image);
//  __TOC__(FACE_QUALITY5PT_SET_IMG)
//
//  __TIC__(FACE_QUALITY5PT_DPU)
//  configurable_dpu_task_->run(0);
//  __TOC__(FACE_QUALITY5PT_DPU)
//  
//  auto ret = vitis::ai::face_quality5pt_post_process_original(
//         configurable_dpu_task_->getInputTensor(),
//         configurable_dpu_task_->getOutputTensor(),
//         configurable_dpu_task_->getConfig());
//  __TOC__(FACE_QUALITY5PT_E2E)
//  return ret[0];
//}

std::vector<FaceQuality5ptResult> FaceQuality5ptImp::run(const std::vector<cv::Mat> &input_images) {
  __TIC__(FACE_QUALITY5PT_E2E)
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  for (auto i = 0u; i < input_images.size(); i++) {
    if (size != input_images[i].size()) {
      cv::Mat img;
      cv::resize(input_images[i], img, size, 0);
      images.push_back(img);
    } else {
      images.push_back(input_images[i]);
    }
  }
  __TIC__(FACE_QUALITY5PT_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(FACE_QUALITY5PT_SET_IMG)

  __TIC__(FACE_QUALITY5PT_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FACE_QUALITY5PT_DPU)
  
  if (mode_ == FaceQuality5pt::Mode::NIGHT) {
     auto ret = vitis::ai::face_quality5pt_post_process(
            configurable_dpu_task_->getInputTensor(),
            configurable_dpu_task_->getOutputTensor(),
            configurable_dpu_task_->getConfig(), false);
  __TOC__(FACE_QUALITY5PT_E2E)
     return ret;

  } else {
    auto ret = vitis::ai::face_quality5pt_post_process(
           configurable_dpu_task_->getInputTensor(),
           configurable_dpu_task_->getOutputTensor(),
           configurable_dpu_task_->getConfig(), true);
  __TOC__(FACE_QUALITY5PT_E2E)
    return ret;
  }
}

//std::vector<FaceQuality5ptResult> FaceQuality5ptImp::run_original(const std::vector<cv::Mat> &input_images) {
//  __TIC__(FACE_QUALITY5PT_E2E)
//  std::vector<cv::Mat> images;
//  auto size = cv::Size(getInputWidth(), getInputHeight());
//  for (auto i = 0u; i < input_images.size(); i++) {
//    if (size != input_images[i].size()) {
//      cv::Mat img;
//      cv::resize(input_images[i], img, size, 0);
//      images.push_back(img);
//    } else {
//      images.push_back(input_images[i]);
//    }
//  }
//  __TIC__(FACE_QUALITY5PT_SET_IMG)
//  configurable_dpu_task_->setInputImageBGR(images);
//  __TOC__(FACE_QUALITY5PT_SET_IMG)
//
//  __TIC__(FACE_QUALITY5PT_DPU)
//  configurable_dpu_task_->run(0);
//  __TOC__(FACE_QUALITY5PT_DPU)
//  
//  auto ret = vitis::ai::face_quality5pt_post_process_original(
//       configurable_dpu_task_->getInputTensor(),
//       configurable_dpu_task_->getOutputTensor(),
//       configurable_dpu_task_->getConfig());
//  __TOC__(FACE_QUALITY5PT_E2E)
//  return ret;
//}


}
}
