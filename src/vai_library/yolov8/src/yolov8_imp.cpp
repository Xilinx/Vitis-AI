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
#include "./yolov8_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;

DEF_ENV_PARAM(ENABLE_YOLOv8_DEBUG, "0");

namespace vitis {
namespace ai {

static void image_preprocess(const cv::Mat input_image, cv::Mat& output_image,
                             const int height, const int width, float& scale,
                             int& left, int& top) {
  cv::Mat image_tmp;

  scale = std::min(float(width) / input_image.cols,
                   float(height) / input_image.rows);
  scale = std::min(scale, 1.0f);
  int unpad_w = round(input_image.cols * scale);
  int unpad_h = round(input_image.rows * scale);
  image_tmp = input_image.clone();

  if (input_image.size() != cv::Size(unpad_w, unpad_h)) {
    cv::resize(input_image, image_tmp, cv::Size(unpad_w, unpad_h),
               cv::INTER_LINEAR);
  }

  float dw = (width - unpad_w) / 2.0f;
  float dh = (height - unpad_h) / 2.0f;

  top = round(dh - 0.1);
  int bottom = round(dh + 0.1);
  left = round(dw - 0.1);
  int right = round(dw + 0.1);

  cv::copyMakeBorder(image_tmp, output_image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  return;
}

YOLOv8Imp::YOLOv8Imp(const std::string& model_name, bool need_preprocess)
    : YOLOv8(model_name, need_preprocess) {}
YOLOv8Imp::YOLOv8Imp(const std::string& model_name, xir::Attrs* attrs,
                     bool need_preprocess)
    : YOLOv8(model_name, attrs, need_preprocess) {}

YOLOv8Imp::~YOLOv8Imp() {}

YOLOv8Result YOLOv8Imp::run(const cv::Mat& input_images) {
  return run(vector<cv::Mat>(1, input_images))[0];
}

vector<YOLOv8Result> YOLOv8Imp::run(const vector<cv::Mat>& input_images) {
  __TIC__(PRE)
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<cv::Mat> images(input_images.size());
  vector<YOLOv8Result> ret;

  vector<float> scales(input_images.size());
  vector<int> left_padding(input_images.size());
  vector<int> top_padding(input_images.size());
  __TIC__(YOLOv8_RESIZE)
  for (auto i = 0u; i < input_images.size(); i++) {
    image_preprocess(input_images[i], images[i], sHeight, sWidth, scales[i],
                     left_padding[i], top_padding[i]);
  }
  __TOC__(YOLOv8_RESIZE)

  __TIC__(YOLOv8_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(YOLOv8_SET_IMG)

  __TOC__(PRE)

  __TIC__(YOLOv8_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(YOLOv8_DPU)

  __TIC__(YOLOv8_POST_ARM)
  ret = vitis::ai::yolov8_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), scales, left_padding, top_padding);
  __TOC__(YOLOv8_POST_ARM)

  return ret;
}

}  // namespace ai
}  // namespace vitis

