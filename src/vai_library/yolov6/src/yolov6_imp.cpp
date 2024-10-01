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
#include "./yolov6_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;

DEF_ENV_PARAM(ENABLE_YOLOV6_DEBUG, "0");
// DEF_ENV_PARAM(DEBUG_YOLOV6_WODFL, "0");

namespace vitis {
namespace ai {

void YOLOv6Imp::letterbox(const cv::Mat& im, int w, int h, cv::Mat& om,
                          float& scale) {
  scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  cv::Mat img_res;
  if (im.size() != cv::Size(w, h)) {
    cv::resize(im, img_res, cv::Size(im.cols * scale, im.rows * scale), 0, 0,
               cv::INTER_LINEAR);
    auto dw = w - img_res.cols;
    auto dh = h - img_res.rows;

    // auto dw = (w - img_res.cols) / 2;
    // auto dh = (h - img_res.rows) / 2;
    if (dw > 0 || dh > 0) {
      om = cv::Mat(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
      copyMakeBorder(img_res, om, 0, dh, 0, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
      // copyMakeBorder(img_res, om, dh, h - dh, dw, w - dw,
      // cv::BORDER_CONSTANT,
      //               cv::Scalar(114, 114, 114));
    } else {
      om = img_res;
    }
  } else {
    om = im;
    scale = 1.0;
  }
}

void YOLOv6Imp::letterbox(const cv::Mat& im, int w, int h, int load_size,
                          cv::Mat& om, float& scale, int& left, int& top) {
  cv::Mat img_res;
  float r = (float)load_size / (float)max(im.cols, im.rows);
  if (r != 1) {
    cv::resize(im, img_res, cv::Size(im.cols * r, im.rows * r), 0, 0,
               r < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
  }
  float dw = (w - img_res.cols) / 2.0f;
  float dh = (h - img_res.rows) / 2.0f;
  top = int(round(dh - 0.1));
  int bottom = int(round(dh + 0.1));
  left = int(round(dw - 0.1));
  int right = int(round(dw + 0.1));
  cv::copyMakeBorder(img_res, om, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  scale = r;
}

YOLOv6Imp::YOLOv6Imp(const std::string& model_name, bool need_preprocess)
    : YOLOv6(model_name, need_preprocess) {}
YOLOv6Imp::YOLOv6Imp(const std::string& model_name, xir::Attrs* attrs,
                     bool need_preprocess)
    : YOLOv6(model_name, attrs, need_preprocess) {}

YOLOv6Imp::~YOLOv6Imp() {}

YOLOv6Result YOLOv6Imp::run(const cv::Mat& input_images) {
  return run(vector<cv::Mat>(1, input_images))[0];
}

vector<YOLOv6Result> YOLOv6Imp::run(const vector<cv::Mat>& input_images) {
  auto use_graph_runner =
      configurable_dpu_task_->getConfig().use_graph_runner();
  auto without_dfl =
      configurable_dpu_task_->getConfig().yolo_v6_param().without_dfl();
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<cv::Mat> images(input_images.size());
  vector<float> scale(input_images.size());

  int force_load_size = 634;
  vector<int> left_padding(input_images.size());
  vector<int> top_padding(input_images.size());

  vector<YOLOv6Result> ret;
  __TIC__(YOLOV5_PRE_ARM)
  for (auto i = 0u; i < input_images.size(); i++) {
    // letterbox(input_images[i], sWidth, sHeight, images[i], scale[i]);
    letterbox(input_images[i], sWidth, sHeight, force_load_size, images[i],
              scale[i], left_padding[i], top_padding[i]);
  }
  __TOC__(YOLOV5_PRE_ARM)

  __TIC__(YOLOV5_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(YOLOV5_SET_IMG)

  // if (use_graph_runner || (ENV_PARAM(DEBUG_YOLOV6_WODFL))) {
  if (use_graph_runner || without_dfl) {
    __TIC__(YOLOV5_DPU)
    configurable_dpu_task_->run(0);
    __TOC__(YOLOV5_DPU)
    __TIC__(YOLOV5_POST_ARM)
    ret = vitis::ai::yolov6_post_process(
        configurable_dpu_task_->getInputTensor()[0],
        configurable_dpu_task_->getOutputTensor()[0],
        configurable_dpu_task_->getConfig(), scale, left_padding, top_padding);
    __TOC__(YOLOV5_POST_ARM)

  } else {
    __TIC__(YOLOV5_DPU_0)
    configurable_dpu_task_->run(0);
    __TOC__(YOLOV5_DPU_0)

    __TIC__(YOLOV5_MIDDLE)
    auto dpu_0_output_tensors = configurable_dpu_task_->getOutputTensor()[0];
    auto size = configurable_dpu_task_->getInputTensor().size();
    std::vector<vitis::ai::library::OutputTensor> output_tensors;
    for (auto j = 0u; j < dpu_0_output_tensors.size(); ++j) {
      if (std::string::npos == dpu_0_output_tensors[j].name.find("reg")) {
        output_tensors.push_back(dpu_0_output_tensors[j]);
        if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
          LOG(INFO) << "dpu output cls:" << dpu_0_output_tensors[j];
        }
        continue;
      }
      for (auto i = 1u; i < size; ++i) {
        int input_w = configurable_dpu_task_->getInputTensor()[i][0].width;
        int w = dpu_0_output_tensors[j].width;
        int h = dpu_0_output_tensors[j].height;
        if (input_w != w * h) {
          continue;
        }
        if (ENV_PARAM(ENABLE_YOLOV6_DEBUG)) {
          LOG(INFO) << "dpu output reg:"
                    << configurable_dpu_task_->getOutputTensor()[i][0];
        }
        __TIC__(YOLOV5_MIDDLE_ARM)
        vitis::ai::yolov6_middle_process(
            configurable_dpu_task_->getInputTensor()[i][0],
            dpu_0_output_tensors[j]);
        __TOC__(YOLOV5_MIDDLE_ARM)
        break;
      }
    }

    for (auto i = 1u; i < size; ++i) {
      __TIC__(YOLOV5_MIDDLE_DPU)
      configurable_dpu_task_->run(i);
      __TOC__(YOLOV5_MIDDLE_DPU)
      output_tensors.push_back(configurable_dpu_task_->getOutputTensor()[i][0]);
    }
    __TOC__(YOLOV5_MIDDLE)

    __TIC__(YOLOV5_POST_ARM)
    // ret = vitis::ai::yolov6_post_process(
    //    configurable_dpu_task_->getInputTensor()[0], output_tensors,
    //    configurable_dpu_task_->getConfig(), scale);

    ret = vitis::ai::yolov6_post_process(
        configurable_dpu_task_->getInputTensor()[0], output_tensors,
        configurable_dpu_task_->getConfig(), scale, left_padding, top_padding);
    __TOC__(YOLOV5_POST_ARM)
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
