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
#include "retinaface_imp.hpp"

#include <vitis/ai/profiling.hpp>

using std::vector;

namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_RETINAFACE_DEBUG, "0");

RetinaFaceImp::RetinaFaceImp(const std::string& model_name,
                             bool need_preprocess)
    : RetinaFace(model_name, need_preprocess),
      processor_{vitis::ai::RetinaFacePostProcess::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig())} {}

RetinaFaceImp::RetinaFaceImp(const std::string& model_name, xir::Attrs* attrs,
                             bool need_preprocess)
    : RetinaFace(model_name, attrs, need_preprocess),

      processor_{vitis::ai::RetinaFacePostProcess::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig())} {}

RetinaFaceImp::~RetinaFaceImp() {}

RetinaFaceResult RetinaFaceImp::run(const cv::Mat& input_image) {
  __TIC__(RETINAFACE_E2E)
  // Set input image into DPU Task
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0);
  } else {
    image = input_image;
  }
  __TIC__(RETINAFACE_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(RETINAFACE_SET_IMG)

  __TIC__(RETINAFACE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(RETINAFACE_DPU)

  __TIC__(RETINAFACE_POST_ARM)
  auto ret = processor_->retinaface_post_process(1u);
  __TOC__(RETINAFACE_POST_ARM)
  __TOC__(RETINAFACE_E2E)
  return ret[0];
}

std::vector<RetinaFaceResult> RetinaFaceImp::run(
    const std::vector<cv::Mat>& input_images) {
  __TIC__(RETINAFACE_E2E)
  // Set input image into DPU Task
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
  __TIC__(RETINAFACE_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(RETINAFACE_SET_IMG)

  __TIC__(RETINAFACE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(RETINAFACE_DPU)

  __TIC__(RETINAFACE_POST_ARM)
  auto ret = processor_->retinaface_post_process(input_images.size());
  __TOC__(RETINAFACE_POST_ARM)

  __TOC__(RETINAFACE_E2E)
  return ret;
}

std::vector<RetinaFaceResult> RetinaFaceImp::run(
    const std::vector<vart::xrt_bo_t>& input_bos) {
  __TIC__(RETINAFACE_E2E)

  __TIC__(RETINAFACE_DPU)
  configurable_dpu_task_->run_with_xrt_bo(input_bos);
  __TOC__(RETINAFACE_DPU)

  __TIC__(RETINAFACE_POST_ARM)
  auto ret = processor_->retinaface_post_process(input_bos.size());
  __TOC__(RETINAFACE_POST_ARM)

  __TOC__(RETINAFACE_E2E)
  return ret;
}

}  // namespace ai
}  // namespace vitis
