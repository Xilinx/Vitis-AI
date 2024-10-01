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
#include "./ofa_yolo_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

#include "utils.hpp"

using namespace std;
namespace vitis {
namespace ai {

OFAYOLOImp::OFAYOLOImp(const std::string& model_name, bool need_preprocess)
    : OFAYOLO(model_name, need_preprocess) {}

OFAYOLOImp::~OFAYOLOImp() {}

OFAYOLOResult OFAYOLOImp::run(const cv::Mat& input_images) {
  return run(vector<cv::Mat>(1, input_images))[0];
}
vector<OFAYOLOResult> OFAYOLOImp::run(const vector<cv::Mat>& input_images) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<cv::Mat> images;
  vector<int> cols, rows;

  __TIC__(OFAYOLO_PRE_ARM)
  for (auto& input_image : input_images) {
    images.push_back(ofa_yolo::letterbox(input_image, sWidth, sHeight).clone());
    cols.push_back(input_image.cols);
    rows.push_back(input_image.rows);
  }
  __TOC__(OFAYOLO_PRE_ARM)

  __TIC__(OFAYOLO_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(OFAYOLO_SET_IMG)

  __TIC__(OFAYOLO_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(OFAYOLO_DPU)

  __TIC__(OFAYOLO_POST_ARM)
  auto ret = vitis::ai::ofa_yolo_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), cols, rows);
  __TOC__(OFAYOLO_POST_ARM)
  return ret;
}

}  // namespace ai
}  // namespace vitis
