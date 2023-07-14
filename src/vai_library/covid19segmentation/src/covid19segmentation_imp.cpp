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

#include "./covid19segmentation_imp.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

namespace vitis {
namespace ai {

using namespace std;
using namespace cv;

static int find_sub(
    const std::string& output,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>& outputs) {
  auto ret = -1;
  for (auto i = 0u; i < outputs.size(); ++i) {
    for (auto j = 0u; j < outputs[i].size(); ++j) {
      if (outputs[i][j].name.find(output) != std::string::npos) {
        ret = j;
        return ret;
      }
    }
  }
  LOG(FATAL) << "cannot found output tensor: " << output;
  return -1;
}

Covid19SegmentationImp::Covid19SegmentationImp(const std::string& model_name,
                                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Covid19Segmentation>(model_name,
                                                           need_preprocess) {}

Covid19SegmentationImp::~Covid19SegmentationImp() {}

Covid19SegmentationResult Covid19SegmentationImp::run_8UC1(
    const cv::Mat& input_image) {
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
  auto sub_sc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[0],
                         configurable_dpu_task_->getOutputTensor());
  auto sub_mc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[1],
                         configurable_dpu_task_->getOutputTensor());
  auto result_sc = segmentation_post_process_8UC1(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_sc]);
  auto result_mc = segmentation_post_process_8UC1(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_mc]);
  __TOC__(post)

  return Covid19SegmentationResult{getInputWidth(), getInputHeight(),
                                   result_sc[0].segmentation,
                                   result_mc[0].segmentation};
}

std::vector<Covid19SegmentationResult> Covid19SegmentationImp::run_8UC1(
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
  auto sub_sc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[0],
                         configurable_dpu_task_->getOutputTensor());
  auto sub_mc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[1],
                         configurable_dpu_task_->getOutputTensor());
  auto result_sc = segmentation_post_process_8UC1(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_sc]);
  auto result_mc = segmentation_post_process_8UC1(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_mc]);
  std::vector<Covid19SegmentationResult> seg_results;
  for (size_t i = 0; i < result_sc.size(); i++) {
    seg_results.push_back(Covid19SegmentationResult{
        getInputWidth(), getInputHeight(), result_sc[i].segmentation,
        result_mc[i].segmentation});
  }
  __TOC__(post)
  return seg_results;
}

Covid19SegmentationResult Covid19SegmentationImp::run_8UC3(
    const cv::Mat& input_image) {
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
  auto sub_sc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[0],
                         configurable_dpu_task_->getOutputTensor());
  auto sub_mc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[1],
                         configurable_dpu_task_->getOutputTensor());
  auto result_sc = segmentation_post_process_8UC3(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_sc],
      configurable_dpu_task_->getConfig());
  auto result_mc = segmentation_post_process_8UC3(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_mc],
      configurable_dpu_task_->getConfig());
  __TOC__(post)

  return Covid19SegmentationResult{getInputWidth(), getInputHeight(),
                                   result_sc[0].segmentation,
                                   result_mc[0].segmentation};
}

std::vector<Covid19SegmentationResult> Covid19SegmentationImp::run_8UC3(
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
  auto sub_sc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[0],
                         configurable_dpu_task_->getOutputTensor());
  auto sub_mc = find_sub(configurable_dpu_task_->getConfig()
                             .platenum_param()
                             .output_tensor_name()[1],
                         configurable_dpu_task_->getOutputTensor());
  auto result_sc = segmentation_post_process_8UC3(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_sc],
      configurable_dpu_task_->getConfig());
  auto result_mc = segmentation_post_process_8UC3(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][sub_mc],
      configurable_dpu_task_->getConfig());
  std::vector<Covid19SegmentationResult> seg_results;
  for (size_t i = 0; i < result_sc.size(); i++) {
    seg_results.push_back(Covid19SegmentationResult{
        getInputWidth(), getInputHeight(), result_sc[i].segmentation,
        result_mc[i].segmentation});
  }
  __TOC__(post)
  return seg_results;
}

}  // namespace ai
}  // namespace vitis
