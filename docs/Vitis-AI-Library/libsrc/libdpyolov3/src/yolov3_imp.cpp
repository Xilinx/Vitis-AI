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
#define ENABLE_NEON
#include "./yolov3_imp.hpp"
#include "utils.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <xilinx/ai/profiling.hpp>

using namespace std;
namespace xilinx {
namespace ai {

YOLOv3Imp::YOLOv3Imp(const std::string &model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<YOLOv3>(model_name, need_preprocess),
      tf_flag_(configurable_dpu_task_->getConfig().is_tf()) {
}

YOLOv3Imp::~YOLOv3Imp() {}

YOLOv3Result YOLOv3Imp::run(const cv::Mat &input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto mAP = configurable_dpu_task_->getConfig().yolo_v3_param().test_map();
  LOG_IF(INFO, false) << "tf_flag_ " << tf_flag_ << " " //
                      << "mAp " << mAP << " "           //
                      << std::endl;
  if (mAP) {
    if (!tf_flag_) {
      int channel = configurable_dpu_task_->getInputTensor()[0][0].channel;
      float scale = xilinx::ai::tensor_scale(
          configurable_dpu_task_->getInputTensor()[0][0]);
      int8_t *data =
          (int8_t *)configurable_dpu_task_->getInputTensor()[0][0].data;
      LOG_IF(INFO, false) << "scale " << scale << " "     //
                          << "sWidth " << sWidth << " "   //
                          << "sHeight " << sHeight << " " //
                          << std::endl;
      yolov3::convertInputImage(input_image, sWidth, sHeight, channel, scale, data);
    } else {
      image = yolov3::letterbox_tf(input_image, sWidth, sHeight).clone();
      configurable_dpu_task_->setInputImageRGB(image);
    }
  } else {
    auto size = cv::Size(sWidth, sHeight);
    if (size != input_image.size()) {
      cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
    } else {
      image = input_image;
    }
    //  convert_RGB(image);
    __TIC__(YOLOV3_SET_IMG)
    configurable_dpu_task_->setInputImageRGB(image);
    __TOC__(YOLOV3_SET_IMG)
  }
  __TIC__(YOLOV3_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(YOLOV3_DPU)

  __TIC__(YOLOV3_POST_ARM)

  auto ret = xilinx::ai::yolov3_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), input_image.cols, input_image.rows);

  __TOC__(YOLOV3_POST_ARM)
  return ret;
}

} // namespace ai
} // namespace xilinx
