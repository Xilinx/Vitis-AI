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
#include "./yolov2_imp.hpp"
#include "utils.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <xilinx/ai/env_config.hpp>
#include <xilinx/ai/profiling.hpp>
using namespace std;
namespace xilinx {
namespace ai {
DEF_ENV_PARAM(ENABLE_YOLOV2_DEBUG, "0");

YOLOv2Imp::YOLOv2Imp(const std::string &model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<YOLOv2>(model_name, need_preprocess) {}

YOLOv2Imp::~YOLOv2Imp() {}

YOLOv2Result YOLOv2Imp::run(const cv::Mat &input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto mAP = configurable_dpu_task_->getConfig().yolo_v3_param().test_map();
  if (mAP) {
    int channel = configurable_dpu_task_->getInputTensor()[0][0].channel;
    float scale = xilinx::ai::tensor_scale(
        configurable_dpu_task_->getInputTensor()[0][0]);
    int8_t *data =
        (int8_t *)configurable_dpu_task_->getInputTensor()[0][0].data;
    yolov2::convertInputImage(input_image, sWidth, sHeight, channel, scale, data);
  } else {
    auto size = cv::Size(sWidth, sHeight);
    if (size != input_image.size()) {
      cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
    } else {
      image = input_image;
    }
    __TIC__(YOLOV2_SET_IMG)
    configurable_dpu_task_->setInputImageRGB(image);
    __TOC__(YOLOV2_SET_IMG)
  }

  __TIC__(YOLOV2_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(YOLOV2_DPU)

  __TIC__(YOLOV2_POST_ARM)
  auto ret = xilinx::ai::yolov2_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), input_image.cols, input_image.rows);
  __TOC__(YOLOV2_POST_ARM)
  return ret;
}

} // namespace ai
} // namespace xilinx
