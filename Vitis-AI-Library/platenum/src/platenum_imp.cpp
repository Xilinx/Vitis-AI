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
#include "./platenum_imp.hpp"
#include <vector>
#include <iostream>
#include <vitis/softmax.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/globalavepool.hpp>
#include <vitis/ai/profiling.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
namespace vitis {
namespace ai {

PlateNumImp::PlateNumImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<PlateNum>(model_name, need_preprocess)
{}

PlateNumImp::~PlateNumImp() {}



PlateNumResult PlateNumImp::run(const cv::Mat &input_image) {
  cv::Mat image;
  auto size = cv::Size(configurable_dpu_task_->getInputTensor()[0][0].width, configurable_dpu_task_->getInputTensor()[0][0].height);
  if (size  != input_image.size()) {
    cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  __TIC__(PLATENUM_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(PLATENUM_SET_IMG)

  __TIC__(PLATENUM_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(PLATENUM_DPU)
	__TIC__(PLATENUM_AVG)
  vitis::ai::globalAvePool((int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(0), 1024, 9, 3,
                               (int8_t*)configurable_dpu_task_->getInputTensor()[1][0].get_data(0), 4);
	__TOC__(PLATENUM_AVG)
	__TIC__(PLATENUM_DPU2)
  configurable_dpu_task_->run(1);
	__TOC__(PLATENUM_DPU2)
  return plate_num_post_process(configurable_dpu_task_->getInputTensor(), configurable_dpu_task_->getOutputTensor())[0];
}

std::vector<PlateNumResult> PlateNumImp::run(const std::vector<cv::Mat>& imgs) {
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
  __TIC__(PLATENUM_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(PLATENUM_SET_IMG)

  __TIC__(PLATENUM_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(PLATENUM_DPU)
	__TIC__(PLATENUM_AVG)
  vitis::ai::globalAvePool((int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(0), 1024, 9, 3,
                               (int8_t*)configurable_dpu_task_->getInputTensor()[1][0].get_data(0), 4);
	__TOC__(PLATENUM_AVG)
	__TIC__(PLATENUM_DPU2)
  configurable_dpu_task_->run(1);
	__TOC__(PLATENUM_DPU2)
  return plate_num_post_process(configurable_dpu_task_->getInputTensor(), configurable_dpu_task_->getOutputTensor());
}

}
}
