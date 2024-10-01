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
#include "./yolovx_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>
using namespace std;
namespace vitis {
namespace ai {

void letterbox(const cv::Mat& im, int w, int h, cv::Mat& om, float& scale) {
  scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  cv::Mat img_res;
  if (im.size() != cv::Size(w, h)) {
    cv::resize(im, img_res, cv::Size(im.cols * scale, im.rows * scale), 0, 0,
               cv::INTER_LINEAR);
    auto dw = w - img_res.cols;
    auto dh = h - img_res.rows;
    if (dw > 0 || dh > 0) {
      om = cv::Mat(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
      copyMakeBorder(img_res, om, 0, dh, 0, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
    } else {
      om = img_res;
    }
  } else {
    om = im;
    scale = 1.0;
  }
}

YOLOvXImp::YOLOvXImp(const std::string& model_name, bool need_preprocess)
    : YOLOvX(model_name, need_preprocess) {}
YOLOvXImp::YOLOvXImp(const std::string& model_name, xir::Attrs* attrs,
                     bool need_preprocess)
    : YOLOvX(model_name, attrs, need_preprocess) {}

YOLOvXImp::~YOLOvXImp() {}

YOLOvXResult YOLOvXImp::run(const cv::Mat& input_images) {
  return run(vector<cv::Mat>(1, input_images))[0];
}
vector<YOLOvXResult> YOLOvXImp::run(const vector<cv::Mat>& input_images) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<cv::Mat> images(input_images.size());
  vector<float> scale(input_images.size());

  __TIC__(YOLOV5_PRE_ARM)
  for (auto i = 0u; i < input_images.size(); i++) {
    letterbox(input_images[i], sWidth, sHeight, images[i], scale[i]);
  }
  __TOC__(YOLOV5_PRE_ARM)

  __TIC__(YOLOV5_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(YOLOV5_SET_IMG)

  __TIC__(YOLOV5_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(YOLOV5_DPU)

  __TIC__(YOLOV5_POST_ARM)
  auto ret = vitis::ai::yolovx_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), scale);

  __TOC__(YOLOV5_POST_ARM)
  return ret;
}

}  // namespace ai
}  // namespace vitis
