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
#include "./openpose_imp.hpp"
#include <vector>
#include <xilinx/ai/env_config.hpp>
#include <xilinx/ai/profiling.hpp>

using namespace std;
namespace xilinx {
namespace ai {

OpenPoseImp::OpenPoseImp(const std::string &model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<OpenPose>(model_name, need_preprocess) {}

OpenPoseImp::~OpenPoseImp() {}

OpenPoseResult OpenPoseImp::run(const cv::Mat &input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  __TIC__(OPENPOSE_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(OPENPOSE_SET_IMG)
  __TIC__(OPENPOSE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(OPENPOSE_DPU)
  __TIC__(OPENPOSE_POST_PROCESS)
  auto ret = xilinx::ai::open_pose_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig(), input_image.cols, input_image.rows);
  __TOC__(OPENPOSE_POST_PROCESS)
  return ret;
}

} // namespace ai
} // namespace xilinx
