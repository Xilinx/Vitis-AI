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
#include "./yolov7_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;

DEF_ENV_PARAM(ENABLE_YOLOv7_DEBUG, "0");

namespace vitis {
namespace ai {

static cv::Mat image_preprocess(const cv::Mat im, const int h, const int w) {
  float scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  int new_w = im.cols * scale;
  int new_h = im.rows * scale;
  cv::Mat img_res;
  cv::resize(im, img_res, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat new_img(cv::Size(w, h), CV_8UC3, cv::Scalar(114, 114, 114));
  int x = (w - new_w) / 2;
  int y = (h - new_h) / 2;
  auto rect = cv::Rect{x, y, new_w, new_h};
  img_res.copyTo(new_img(rect));
  return new_img;
}

YOLOv7Imp::YOLOv7Imp(const std::string& model_name, bool need_preprocess)
    : YOLOv7(model_name, need_preprocess) {}


YOLOv7Imp::~YOLOv7Imp() {}

YOLOv7Result YOLOv7Imp::run(const cv::Mat& input_images) {
  return run(vector<cv::Mat>(1, input_images))[0];
}

vector<YOLOv7Result> YOLOv7Imp::run(const vector<cv::Mat>& input_images) {
  __TIC__(PREPROCESS)
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<cv::Mat> images(input_images.size());
  __TIC__(YOLOv7_RESIZE)
  for (auto i = 0u; i < input_images.size(); i++) {
    images[i]=image_preprocess(input_images[i], sHeight, sWidth);
  }
  __TOC__(YOLOv7_RESIZE)

  __TIC__(YOLOv7_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(YOLOv7_SET_IMG)

  __TOC__(PREPROCESS)

  __TIC__(YOLOv7_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(YOLOv7_DPU)
  vector<int> ws, hs;
  for (auto input_image : input_images) {
    ws.push_back(input_image.cols);
    hs.push_back(input_image.rows);
  }
  __TIC__(YOLOv7_POST_ARM)
  auto ret = vitis::ai::yolov7_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), ws, hs);
  __TOC__(YOLOv7_POST_ARM)

  return ret;
}

}  // namespace ai
}  // namespace vitis
