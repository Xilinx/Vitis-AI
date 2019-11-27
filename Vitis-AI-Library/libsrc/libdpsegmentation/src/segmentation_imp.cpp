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

#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <xilinx/ai/profiling.hpp>
#include "./segmentation_imp.hpp"

namespace  xilinx {namespace ai {

using namespace std;
using namespace cv;

SegmentationImp::SegmentationImp(const std::string& model_name, bool need_preprocess) :
  xilinx::ai::TConfigurableDpuTask<Segmentation>(model_name, need_preprocess)
{

}

SegmentationImp::~SegmentationImp() {
}


SegmentationResult SegmentationImp::run_8UC1(const cv::Mat& input_image)
{
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0,0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  __TIC__(SEGMENTATION_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(SEGMENTATION_SET_IMG)

  __TIC__(SEGMENTATION_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SEGMENTATION_DPU)

  __TIC__(post)
  auto result = segmentation_post_process_8UC1(configurable_dpu_task_->getInputTensor()[0],
         configurable_dpu_task_->getOutputTensor()[0]);
  __TOC__(post)

  return result;
}


SegmentationResult SegmentationImp::run_8UC3(const cv::Mat& input_image)
{
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, cv::INTER_NEAREST);
  } else {
    image = input_image;
  }
   __TIC__(SEGMENTATION_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(SEGMENTATION_SET_IMG)

  __TIC__(SEGMENTATION_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SEGMENTATION_DPU)

  __TIC__(post)
  auto result = segmentation_post_process_8UC3(configurable_dpu_task_->getInputTensor()[0],
         configurable_dpu_task_->getOutputTensor()[0], configurable_dpu_task_->getConfig());
  __TOC__(post)
  return result;
}


}}
