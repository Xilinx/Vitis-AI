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

#include "./segmentation_imp.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

namespace vitis {
namespace ai {

using namespace std;
using namespace cv;

DEF_ENV_PARAM(ENABLE_SEGMENTATION_DEBUG, "0");

SegmentationImp::SegmentationImp(const std::string& model_name,
                                 bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Segmentation>(model_name,
                                                    need_preprocess) {
  output_tensor_index_ = 0;
  auto specify_output = configurable_dpu_task_->getConfig()
                            .segmentation_param()
                            .specify_output_layer();
  LOG_IF(INFO, ENV_PARAM(ENABLE_SEGMENTATION_DEBUG))
      << "specify_output :" << specify_output;
  if (specify_output) {
    auto key = configurable_dpu_task_->getConfig()
                   .segmentation_param()
                   .output_tensor_name();
    LOG_IF(INFO, ENV_PARAM(ENABLE_SEGMENTATION_DEBUG))
        << "size:" << configurable_dpu_task_->getOutputTensor()[0].size();
    for (auto i = 0u; i < configurable_dpu_task_->getOutputTensor()[0].size();
         ++i) {
      auto name = configurable_dpu_task_->getOutputTensor()[0][i].name;
      LOG_IF(INFO, ENV_PARAM(ENABLE_SEGMENTATION_DEBUG))
          << i << ", name:" << name;
      if (std::string::npos != name.find(key)) {
        output_tensor_index_ = i;
        LOG_IF(INFO, ENV_PARAM(ENABLE_SEGMENTATION_DEBUG))
            << "find output index:" << i << ", name:" << name;
        break;
      }
    }
  }
  LOG_IF(INFO, ENV_PARAM(ENABLE_SEGMENTATION_DEBUG))
      << "output_tensor_index_:" << output_tensor_index_;
}

SegmentationImp::~SegmentationImp() {}

SegmentationResult SegmentationImp::run_8UC1(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  __TIC__(SEGMENTATION_SET_IMG)
  if (configurable_dpu_task_->getConfig().order_type() == 1) {
    configurable_dpu_task_->setInputImageBGR(image);
  } else if (configurable_dpu_task_->getConfig().order_type() == 2) {
    configurable_dpu_task_->setInputImageRGB(image);
  } else {
    LOG(FATAL) << "unknown image order type";
  }
  __TOC__(SEGMENTATION_SET_IMG)

  __TIC__(SEGMENTATION_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SEGMENTATION_DPU)

  __TIC__(post)
  auto result = segmentation_post_process_8UC1(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][output_tensor_index_]);
  __TOC__(post)

  return result[0];
}

std::vector<SegmentationResult> SegmentationImp::run_8UC1(
    const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  for (auto i = 0u; i < input_images.size(); i++) {
    cv::Mat image;
    if (size != input_images[i].size()) {
      cv::resize(input_images[i], image, size, 0, 0, cv::INTER_LINEAR);
      images.push_back(image);
    } else {
      images.push_back(input_images[i]);
    }
  }
  __TIC__(SEGMENTATION_SET_IMG)
  if (configurable_dpu_task_->getConfig().order_type() == 1) {
    configurable_dpu_task_->setInputImageBGR(images);
  } else if (configurable_dpu_task_->getConfig().order_type() == 2) {
    configurable_dpu_task_->setInputImageRGB(images);
  } else {
    LOG(FATAL) << "unknown image order type";
  }
  __TOC__(SEGMENTATION_SET_IMG)

  __TIC__(SEGMENTATION_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SEGMENTATION_DPU)

  __TIC__(post)
  auto results = segmentation_post_process_8UC1(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][output_tensor_index_]);
  __TOC__(post)

  return results;
}

SegmentationResult SegmentationImp::run_8UC3(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, cv::INTER_NEAREST);
  } else {
    image = input_image;
  }
  __TIC__(SEGMENTATION_SET_IMG)
  if (configurable_dpu_task_->getConfig().order_type() == 1) {
    configurable_dpu_task_->setInputImageBGR(image);
  } else if (configurable_dpu_task_->getConfig().order_type() == 2) {
    configurable_dpu_task_->setInputImageRGB(image);
  } else {
    LOG(FATAL) << "unknown image order type";
  }
  __TOC__(SEGMENTATION_SET_IMG)

  __TIC__(SEGMENTATION_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SEGMENTATION_DPU)

  __TIC__(post)
  auto result = segmentation_post_process_8UC3(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][output_tensor_index_],
      configurable_dpu_task_->getConfig());
  __TOC__(post)
  return result[0];
}

std::vector<SegmentationResult> SegmentationImp::run_8UC3(
    const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  for (auto i = 0u; i < input_images.size(); i++) {
    cv::Mat image;
    if (size != input_images[i].size()) {
      cv::resize(input_images[i], image, size, 0, 0, cv::INTER_LINEAR);
      images.push_back(image);
    } else {
      images.push_back(input_images[i]);
    }
  }

  __TIC__(SEGMENTATION_SET_IMG)
  if (configurable_dpu_task_->getConfig().order_type() == 1) {
    configurable_dpu_task_->setInputImageBGR(images);
  } else if (configurable_dpu_task_->getConfig().order_type() == 2) {
    configurable_dpu_task_->setInputImageRGB(images);
  } else {
    LOG(FATAL) << "unknown image order type";
  }
  __TOC__(SEGMENTATION_SET_IMG)

  __TIC__(SEGMENTATION_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SEGMENTATION_DPU)
  __TIC__(post)
  auto results = segmentation_post_process_8UC3(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][output_tensor_index_],
      configurable_dpu_task_->getConfig());
  __TOC__(post)
  return results;
}

}  // namespace ai
}  // namespace vitis
