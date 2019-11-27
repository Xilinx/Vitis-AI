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
#include "./classification_imp.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <vector>
#include <xilinx/ai/env_config.hpp>
#include <xilinx/ai/profiling.hpp>

using namespace std;
namespace xilinx {
namespace ai {
DEF_ENV_PARAM(ENABLE_CLASSIFICATION_DEBUG, "0");
ClassificationImp::ClassificationImp(const std::string &model_name,
                                     bool need_preprocess)
    : TConfigurableDpuTask<Classification>(model_name, need_preprocess),
      preprocess_type{configurable_dpu_task_->getConfig().classification_param().preprocess_type()},
      TOP_K{configurable_dpu_task_->getConfig().classification_param().top_k()},
      test_accuracy{configurable_dpu_task_->getConfig()
                        .classification_param()
                        .test_accuracy()} {}

ClassificationImp::~ClassificationImp() {}

static void croppedImage(const cv::Mat &image, int height, int width,
                         cv::Mat &cropped_img) {
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
}

static void inception_preprocess (const cv::Mat &image,int height, int width, cv::Mat &pro_res,
                                  float central_fraction = 0.875, bool iscentral_crop=true){
  cv::Mat res_crop = image;
  if (iscentral_crop) {
    float img_hd = image.rows;
    float img_wd = image.cols;
    int offset_h = (img_hd - img_hd * central_fraction) / 2;
    int offset_w = (img_wd - img_wd * central_fraction) / 2;
    cv::Rect box(offset_w, offset_h, image.cols - offset_w * 2,
                 image.rows - offset_h * 2);
    res_crop = image(box).clone();
  }

  if (height && width) {
    cv::resize(res_crop, pro_res, cv::Size(width, height));
  }
}

static void vgg_preprocess(const cv::Mat &image, int height, int width, cv::Mat & pro_res){
  float smallest_side = 256;
  float scale = smallest_side / ((image.rows > image.cols) ? image.cols: image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(image.cols * scale, image.rows * scale));
  croppedImage(resized_image, height, width, pro_res);
}

xilinx::ai::ClassificationResult
ClassificationImp::run(const cv::Mat &input_image) {
  cv::Mat image;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  if (size == input_image.size()) {
    image = input_image;
  } else {
    //::xilinx::ai::proto::ClassificationParam_PreprocessType
    switch(preprocess_type){
    case 0:
      cv::resize(input_image, image, size);
      break;
    case 1:
      if (test_accuracy) {
        croppedImage(input_image, height, width, image);
      } else {
        cv::resize(input_image, image, size);
      }
      break;
    case 2:
      vgg_preprocess(input_image, height, width, image);
      break;
    case 3:
      inception_preprocess(input_image, height, width, image);
      break;
    default:
      break;
    }
  }
  //__TIC__(CLASSIFY_E2E_TIME)
  __TIC__(CLASSIFY_SET_IMG)
  if (preprocess_type == 2 || preprocess_type == 3) {
    configurable_dpu_task_->setInputImageRGB(image);
  } else {
    configurable_dpu_task_->setInputImageBGR(image);
  }
  __TOC__(CLASSIFY_SET_IMG)

  __TIC__(CLASSIFY_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(CLASSIFY_DPU)

  __TIC__(CLASSIFY_POST_ARM)
  auto ret =
      classification_post_process(configurable_dpu_task_->getInputTensor()[0],
                                  configurable_dpu_task_->getOutputTensor()[0],
                                  configurable_dpu_task_->getConfig());

  if (configurable_dpu_task_->getOutputTensor()[0][0].channel == 1001) {
    for (auto &s : ret.scores) {
      s.index--;
    }
  }
  __TOC__(CLASSIFY_POST_ARM)
  //__TOC__(CLASSIFY_E2E_TIME)

  return ret;
}

} // namespace ai
} // namespace xilinx
