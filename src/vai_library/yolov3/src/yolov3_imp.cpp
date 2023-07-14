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
#include "./yolov3_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

#include "utils.hpp"

using namespace std;
namespace vitis {
namespace ai {

YOLOv3Imp::YOLOv3Imp(const std::string& model_name, bool need_preprocess)
    : YOLOv3(model_name, need_preprocess),
      tf_flag_(configurable_dpu_task_->getConfig().is_tf()) {}

YOLOv3Imp::~YOLOv3Imp() {}

YOLOv3Result YOLOv3Imp::run(const cv::Mat& input_images) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto mAP = configurable_dpu_task_->getConfig().yolo_v3_param().test_map();
  auto type = configurable_dpu_task_->getConfig().yolo_v3_param().type();
  LOG_IF(INFO, false) << "tf_flag_ " << tf_flag_ << " "  //
                      << "mAp " << mAP << " "            //
                      << std::endl;
  if (mAP) {
    if (!tf_flag_) {
      int channel = configurable_dpu_task_->getInputTensor()[0][0].channel;
      float scale =
          library::tensor_scale(configurable_dpu_task_->getInputTensor()[0][0]);
//# DPUV1 needs float input data
#ifdef ENABLE_DPUCADX8G_RUNNER
      float* data =
          (float*)configurable_dpu_task_->getInputTensor()[0][0].get_data(0);
#else
      int8_t* data =
          (int8_t*)configurable_dpu_task_->getInputTensor()[0][0].get_data(0);
#endif

      LOG_IF(INFO, false) << "scale " << scale << " "      //
                          << "sWidth " << sWidth << " "    //
                          << "sHeight " << sHeight << " "  //
                          << std::endl;
      yolov3::convertInputImage(input_images, sWidth, sHeight, channel, scale,
                                data);
    } else {
      if ((type == 1) ||
          (type == 2)) {  // 1: yolov4-csp; 2: yolov5-large/yolov5-nano/yolov5s6
        image = yolov3::letterbox(input_images, sWidth, sHeight).clone();
      } else {
        image = yolov3::letterbox_tf(input_images, sWidth, sHeight).clone();
      }
      configurable_dpu_task_->setInputImageRGB(image);
    }
  } else {
    auto size = cv::Size(sWidth, sHeight);
    if (size != input_images.size()) {
      cv::resize(input_images, image, size, 0, 0, cv::INTER_LINEAR);
    } else {
      image = input_images;
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

  auto ret = vitis::ai::yolov3_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), input_images.cols,
      input_images.rows);

  __TOC__(YOLOV3_POST_ARM)
  return ret;
}  // namespace ai

vector<YOLOv3Result> YOLOv3Imp::run(const vector<cv::Mat>& input_images) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto mAP = configurable_dpu_task_->getConfig().yolo_v3_param().test_map();
  auto type = configurable_dpu_task_->getConfig().yolo_v3_param().type();
  LOG_IF(INFO, false) << "tf_flag_ " << tf_flag_ << " "  //
                      << "mAp " << mAP << " "            //
                      << std::endl;
  if (mAP) {
    if (!tf_flag_) {
      int channel = configurable_dpu_task_->getInputTensor()[0][0].channel;
      float scale =
          library::tensor_scale(configurable_dpu_task_->getInputTensor()[0][0]);
      for (size_t i = 0; i < input_images.size(); i++) {
//# DPUV1 needs float input data
#ifdef ENABLE_DPUCADX8G_RUNNER
        float* data =
            (float*)configurable_dpu_task_->getInputTensor()[0][0].get_data(i);
#else
        int8_t* data =
            (int8_t*)configurable_dpu_task_->getInputTensor()[0][0].get_data(i);
#endif
        LOG_IF(INFO, false) << "scale " << scale << " "      //
                            << "sWidth " << sWidth << " "    //
                            << "sHeight " << sHeight << " "  //
                            << std::endl;
        yolov3::convertInputImage(input_images[i], sWidth, sHeight, channel,
                                  scale, data);
      }
    } else {
      vector<cv::Mat> images;
      for (auto input_image : input_images) {
        if (type == 1 ||
            type == 2) {  // 1: yolov4-csp; 2: yolov5-large/yolov5-nano/yolov5s6
          images.push_back(
              yolov3::letterbox(input_image, sWidth, sHeight).clone());
        } else {
          images.push_back(
              yolov3::letterbox_tf(input_image, sWidth, sHeight).clone());
        }
      }
      configurable_dpu_task_->setInputImageRGB(images);
    }
  } else {
    auto size = cv::Size(sWidth, sHeight);
    vector<cv::Mat> images;
    cv::Mat image;
    for (auto& input_image : input_images) {
      if (size != input_image.size()) {
        cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
      } else {
        image = input_image;
      }
      images.push_back(image.clone());
    }
    //  convert_RGB(image);
    __TIC__(YOLOV3_SET_IMG)
    configurable_dpu_task_->setInputImageRGB(images);
    __TOC__(YOLOV3_SET_IMG)
  }
  __TIC__(YOLOV3_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(YOLOV3_DPU)

  __TIC__(YOLOV3_POST_ARM)
  vector<int> cols, rows;
  for (auto input_image : input_images) {
    cols.push_back(input_image.cols);
    rows.push_back(input_image.rows);
  }
  auto ret = vitis::ai::yolov3_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), cols, rows);

  __TOC__(YOLOV3_POST_ARM)
  return ret;
}

vector<YOLOv3Result> YOLOv3Imp::run(const vector<vart::xrt_bo_t>& input_bos) {
  __TIC__(YOLOV3_DPU)
  configurable_dpu_task_->run_with_xrt_bo(input_bos);
  __TOC__(YOLOV3_DPU)

  __TIC__(YOLOV3_POST_ARM)
  auto batch = input_bos.size();
  vector<int> cols(batch, getInputWidth());
  vector<int> rows(batch, getInputHeight());
  auto ret = vitis::ai::yolov3_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), cols, rows);

  __TOC__(YOLOV3_POST_ARM)
  return ret;
}

}  // namespace ai
}  // namespace vitis
