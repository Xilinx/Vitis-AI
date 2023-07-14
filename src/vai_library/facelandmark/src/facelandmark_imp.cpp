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
#include "./facelandmark_imp.hpp"

#include <opencv2/imgproc.hpp>
#include <vitis/ai/profiling.hpp>

using std::vector;

namespace vitis {
namespace ai {

FaceLandmarkImp::FaceLandmarkImp(const std::string &model_name,
                                 bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<FaceLandmark>(model_name,
                                                    need_preprocess) {}

FaceLandmarkImp::FaceLandmarkImp(const std::string &model_name,
                               xir::Attrs *attrs,
                                 bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<FaceLandmark>(model_name,
                                                    attrs,
                                                    need_preprocess) {}
FaceLandmarkImp::~FaceLandmarkImp() {}

FaceLandmarkResult FaceLandmarkImp::run(const cv::Mat &input_image) {
  cv::Mat image;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  __TIC__(FACE_LANDMARK_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(FACE_LANDMARK_SET_IMG)

  __TIC__(FACE_LANDMARK_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FACE_LANDMARK_DPU)

  __TIC__(FACE_LANDMARK_POST_ARM)
  auto ret = vitis::ai::face_landmark_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig());
  __TOC__(FACE_LANDMARK_POST_ARM)

  return ret[0];
}

std::vector<FaceLandmarkResult> FaceLandmarkImp::run(
    const std::vector<cv::Mat> &input_images) {
  std::vector<cv::Mat> images;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  for (auto i = 0u; i < input_images.size(); i++) {
    if (size != input_images[i].size()) {
      cv::Mat img;
      cv::resize(input_images[i], img, size, 0, 0, cv::INTER_LINEAR);
      images.push_back(img);
    } else {
      images.push_back(input_images[i]);
    }
  }

  __TIC__(FACE_LANDMARK_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(FACE_LANDMARK_SET_IMG)

  __TIC__(FACE_LANDMARK_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FACE_LANDMARK_DPU)

  __TIC__(FACE_LANDMARK_POST_ARM)
  auto ret = vitis::ai::face_landmark_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig());
  __TOC__(FACE_LANDMARK_POST_ARM)

  return ret;
}

}  // namespace ai
}  // namespace vitis
