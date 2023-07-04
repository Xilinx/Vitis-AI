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
#include "./openpose_imp.hpp"

#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

OpenPoseImp::OpenPoseImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<OpenPose>(model_name, need_preprocess) {}

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
  auto ret = vitis::ai::open_pose_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), input_image.cols, input_image.rows,
      0);
  __TOC__(OPENPOSE_POST_PROCESS)
  return ret;
}
vector<OpenPoseResult> OpenPoseImp::run(const vector<cv::Mat> &input_images) {
  vector<cv::Mat> images;
  vector<int> ws, hs;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  for (auto& input_image : input_images) {
    Mat image;
    if (size != input_image.size()) {
      cv::resize(input_image, image, size);
    } else {
      image = input_image;
    }
    images.push_back(image);
    ws.push_back(input_image.cols);
    hs.push_back(input_image.rows);
  }
  __TIC__(OPENPOSE_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(OPENPOSE_SET_IMG)
  __TIC__(OPENPOSE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(OPENPOSE_DPU)
  __TIC__(OPENPOSE_POST_PROCESS)
  auto ret = vitis::ai::open_pose_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), ws, hs);
  __TOC__(OPENPOSE_POST_PROCESS)
  return ret;
}

}  // namespace ai
}  // namespace vitis
