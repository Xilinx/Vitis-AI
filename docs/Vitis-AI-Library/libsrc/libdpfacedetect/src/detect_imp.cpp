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
#include "detect_imp.hpp"
#include <xilinx/ai/profiling.hpp>

using std::vector;

namespace xilinx {
namespace ai {

DetectImp::DetectImp(const std::string &model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<FaceDetect>(model_name, need_preprocess),
      det_threshold_(configurable_dpu_task_->getConfig()
                         .dense_box_param()
                         .det_threshold()) { //
}

DetectImp::~DetectImp() {}

FaceDetectResult DetectImp::run(const cv::Mat &input_image) {
  __TIC__(FACE_DETECT_E2E)
  // Set input image into DPU Task
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0);
  } else {
    image = input_image;
  }
  __TIC__(FACE_DETECT_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(FACE_DETECT_SET_IMG)

  __TIC__(FACE_DETECT_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FACE_DETECT_DPU)

  __TIC__(FACE_DETECT_POST_ARM)
  auto ret = xilinx::ai::face_detect_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig(), det_threshold_);
  __TOC__(FACE_DETECT_POST_ARM)

  __TOC__(FACE_DETECT_E2E)
  return ret;
}

float DetectImp::getThreshold() const {
  std::lock_guard<std::mutex> lock(mtx_threshold_);
  return det_threshold_;
}

void DetectImp::setThreshold(float threshold) {
  std::lock_guard<std::mutex> lock(mtx_threshold_);
  det_threshold_ = threshold;
}

} // namespace ai
} // namespace xilinx
