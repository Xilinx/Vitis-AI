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
#include "./facefeature_imp.hpp"

#include <opencv2/imgproc.hpp>
#include <vitis/ai/profiling.hpp>

namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_FACE_FEATURE_DEBUG, "0");
FaceFeatureImp::FaceFeatureImp(const std::string &model_name,
                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<FaceFeature>(model_name,
                                                   need_preprocess) {}

FaceFeatureImp::FaceFeatureImp(const std::string &model_name,
                               xir::Attrs *attrs,
                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<FaceFeature>(model_name,
                                                   attrs,
                                                   need_preprocess) {}


FaceFeatureImp::~FaceFeatureImp() {}

void FaceFeatureImp::run_internal(const cv::Mat &input_image) {
  __TIC__(FACE_FEATURE_IMG_RESIZE)
  cv::Mat image;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  __TOC__(FACE_FEATURE_IMG_RESIZE)
  __TIC__(FACE_FEATURE_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(FACE_FEATURE_SET_IMG)

  __TIC__(FACE_FEATURE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FACE_FEATURE_DPU)
}

void FaceFeatureImp::run_internal(const std::vector<cv::Mat> &input_images) {
  __TIC__(FACE_FEATURE_IMG_RESIZE_BATCH)
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
  __TOC__(FACE_FEATURE_IMG_RESIZE_BATCH)
  __TIC__(FACE_FEATURE_SET_IMG_BATCH)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(FACE_FEATURE_SET_IMG_BATCH)

  __TIC__(FACE_FEATURE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FACE_FEATURE_DPU)
}

FaceFeatureFixedResult FaceFeatureImp::run_fixed(const cv::Mat &input_image) {
  run_internal(input_image);
  __TIC__(FACE_FEATURE_POST_ARM)
  auto result = vitis::ai::face_feature_post_process_fixed(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig());
  __TOC__(FACE_FEATURE_POST_ARM)
  return std::move(result[0]);
}

FaceFeatureFloatResult FaceFeatureImp::run(const cv::Mat &input_image) {
  run_internal(input_image);
  __TIC__(FACE_FEATURE_POST_ARM)
  auto result = vitis::ai::face_feature_post_process_float(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig());
  __TOC__(FACE_FEATURE_POST_ARM)
  return std::move(result[0]);
}

std::vector<FaceFeatureFixedResult> FaceFeatureImp::run_fixed(
    const std::vector<cv::Mat> &input_images) {
  run_internal(input_images);
  __TIC__(FACE_FEATURE_POST_ARM)
  auto result = vitis::ai::face_feature_post_process_fixed(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig());
  __TOC__(FACE_FEATURE_POST_ARM)
  return result;
}

std::vector<FaceFeatureFloatResult> FaceFeatureImp::run(
    const std::vector<cv::Mat> &input_images) {
  run_internal(input_images);
  __TIC__(FACE_FEATURE_POST_ARM)
  auto result = vitis::ai::face_feature_post_process_float(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig());
  __TOC__(FACE_FEATURE_POST_ARM)
  return result;
}

}  // namespace ai
}  // namespace vitis
